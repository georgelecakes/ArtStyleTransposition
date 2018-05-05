# -*- coding: utf-8 -*-
"""
Created on Wed May  2 21:35:32 2018

@author: George
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy.io
import PIL.Image
import collections

def loadContentImage(filename, imageSize):
    image = PIL.Image.open(filename)
    largestImageDimension = np.max(image.size)
    factor = float(imageSize) / largestImageDimension
    size = np.array(image.size) * factor
    size = size.astype(int)
    image.resize(size, PIL.Image.LANCZOS)
    return np.float32(image)
    
def loadStyleImage(filename, contentImageShape):
    image = PIL.Image.open(filename)
    image.resize(contentImageShape, PIL.Image.LANCZOS)
    return np.float32(image)
    
contentImageName = r"images\Arthur.jpg"
styleImageName = r"images\starry-night.jpg"
finalImageFileName = r"Output.jpg"
numberOfIterations = 10000
lossRatio = 1e-3

vggModel = r"pre_trained_model\imagenet-vgg-verydeep-19.mat"
vggData = scipy.io.loadmat(vggModel)
#print(vggData['meta'])   #Contains two keys, 'layers' and 'meta', meta seems to list all the objects that have been learned
vggWeights = vggData['layers'][0]

#VGG19 Layers, necessary to understand what layers I need to use for loss
layers = (
        "conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1",
        "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2",
        "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3","relu3_3","conv3_4","relu3_4","pool3",
        "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3","relu4_3","conv4_4","relu4_4","pool4",
        "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3","relu5_3","conv5_4","relu5_4")


contentImage = loadContentImage(contentImageName, 512)
shape = (contentImage.shape[1],contentImage.shape[0])
styleImage = loadStyleImage(styleImageName, shape)

#initialize random noise image
initialImage = np.random.normal(size = contentImage.shape, scale=np.std(contentImage))
#initialImage = contentImage

#Display all three images
plt.figure(1)
plt.imshow(contentImage/ 255.0, interpolation='sinc')
plt.figure(2)
plt.imshow(styleImage/ 255.0, interpolation='sinc')
plt.figure(3)
plt.imshow(initialImage/ 255.0, interpolation='sinc')

#Important layer information for content and styles, deciding what layers within VGG19 we are going to use for loss
#What layer are we using to get the content information?
contentLayersForLoss = ["conv4_2"]
#What layers are we using to get the style information?
styleLayersForLoss = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]

#Content loss weights, must equal one for each
contentLayerWeight = [1.0]
styleLayerWeight = [0.2, 0.2, 0.2, 0.2, 0.2] #Divided amont the 5 relu levels, can adjust to change amount of influence

#Create maps for layers relating the layer to the weight for content and style
contentLayers = {}
styleLayers = {}

for layer, weight in zip(contentLayersForLoss, contentLayerWeight):
    contentLayers[layer] = weight
    
for layer, weight in zip(styleLayersForLoss, styleLayerWeight):
    styleLayers[layer] = weight

#Create our tensor flow session
#Allow tensorflow to automatiically choose a device in case the one we chose is unavailible
session = tf.Session(config=tf.ConfigProto(allow_soft_placement = True))

#Construct the style transfer system

#Transform images into VGG19 format (batch, height, width, channel)
tempShape = (1,) + contentImage.shape
contentImage = np.reshape(contentImage, tempShape)
tempShape = (1,) + styleImage.shape
styleImage = np.reshape(styleImage, tempShape)
tempShape = (1,) + initialImage.shape
initialImage = np.reshape(initialImage, tempShape)

#Convert images into floating point representation recentered around zero
contentImageFloat = np.float32( contentImage - np.array([128.0, 128.0, 128.0]) )
styleImageFloat = np.float32(styleImage - np.array([128.0, 128.0, 128.0]))
initialImageFloat = np.float32(initialImage - np.array([128.0, 128.0, 128.0]))

#Make a convolution layer
def ConvLayer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)

#Make a max pooling layer
def PoolLayer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')

#Creates the feed forward network for VGG, also does some conversions based on differences
def FeedForward(input_image,vggData,  scope=None):
    net = {}
    current = input_image

    with tf.variable_scope(scope):
        for i, name in enumerate(layers):
            kind = name[:4]
            if kind == 'conv':
                kernels = vggData['layers'][0][i][0][0][2][0][0]
                bias = vggData['layers'][0][i][0][0][2][0][1]

                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                bias = bias.reshape(-1)

                current = ConvLayer(current, kernels, bias)
            elif kind == 'relu':
                current = tf.nn.relu(current)
            elif kind == 'pool':
                current =PoolLayer(current)
            net[name] = current

    assert len(net) == len(layers)
    return net

#Taken from other implementations
def CalculateGramMatrix(tensor):
    shape = tensor.get_shape()
    numChannels = int(shape[3])
    matrix = tf.reshape(tensor, shape=[-1,numChannels])
    gramMatrix = tf.matmul(tf.transpose(matrix), matrix)
    return gramMatrix

#Create the VGG19 Graph
def CreateStyleGraph(vggData, contentLayersAndWeights, styleLayersAndWeights):
    
    pregeneratedContentLayers = collections.OrderedDict(sorted(contentLayersAndWeights.items()))
    pregeneratedStyleLayers = collections.OrderedDict(sorted(styleLayersAndWeights.items()))
    
    #Variable & Placehodlers
    initialImageVariable = tf.Variable(initialImageFloat, trainable=True, dtype=tf.float32)
    styleImagePlaceholder = tf.placeholder(tf.float32, shape = styleImageFloat.shape, name="style" )
    contentImagePlaceholder = tf.placeholder(tf.float32, shape = contentImageFloat.shape, name = "content")
    
    contentLayers = FeedForward(contentImagePlaceholder, vggData, scope="content")
    contentFeatures = {}
    for i in pregeneratedContentLayers:
        contentFeatures[i] = contentLayers[i]
    styleLayers = FeedForward(styleImagePlaceholder, vggData, scope="style")
    styleFeatures = {}
    
    for i in pregeneratedStyleLayers:
        styleFeatures[i] = CalculateGramMatrix(styleLayers[i])
    initialValues = FeedForward( initialImageVariable , vggData, scope = "mixed")
    
    L_content = 0
    L_style = 0
    for id in initialValues:
        if id in pregeneratedContentLayers:
            ## content loss ##

            F = initialValues[id]            # content feature of x
            P = contentFeatures[id]            # content feature of p

            _, h, w, d = F.get_shape() # first return value is batch size (must be one)
            N = h.value*w.value        # product of width and height
            M = d.value                # number of filters

            w = pregeneratedContentLayers[id]# weight for this layer

            L_content += w * tf.reduce_sum(tf.pow((F-P), 2)) / 2 # original paper
            #L_content += w * tf.reduce_sum(tf.pow((F-P), 2)) / (N*M) #artistic style transfer for videos
            #L_content += w * (1. / (2. * np.sqrt(M) * np.sqrt(N))) * tf.reduce_sum(tf.pow((F - P), 2))

        elif id in pregeneratedStyleLayers:
            ## style loss ##

            F = initialValues[id]

            _, h, w, d = F.get_shape()  # first return value is batch size (must be one)
            N = h.value * w.value       # product of width and height
            M = d.value                 # number of filters

            w = pregeneratedStyleLayers[id]   # weight for this layer

            G = CalculateGramMatrix(F)    # style feature of x
            A = styleFeatures[id]             # style feature of a

            print(G.shape)
            print(A.shape)

            L_style += w * (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((G-A), 2))
    
    #Combine to determine total loss
    alphaLoss = lossRatio
    betaLoss = 1.0
    
    contentLoss = L_content
    styleLoss = L_style
    
    totalLoss = alphaLoss * contentLoss + betaLoss * styleLoss
    
    return [initialImageVariable, styleImagePlaceholder, contentImagePlaceholder, contentLoss, styleLoss, totalLoss]

def Update(session, numberOfIterations, initialImageVariable, styleImagePlaceholder, contentImagePlaceholder, contentImageFloat, styleImageFloat, contentLoss, styleLoss, totalLoss):
    # this call back function is called every after loss is updated
    global _iter
    _iter = 0
    def callback(tl, cl, sl):
        global _iter
        print('iter : %4d, ' % _iter, 'L_total : %g, L_content : %g, L_style : %g' % (tl, cl, sl))
        _iter += 1

    optimizer = tf.contrib.opt.ScipyOptimizerInterface(totalLoss, method='L-BFGS-B', options={'maxiter': numberOfIterations})

    # initialize variables
    init_op = tf.global_variables_initializer()
    session.run(init_op)

    # optmization
    optimizer.minimize(session,feed_dict={styleImagePlaceholder:styleImageFloat, contentImagePlaceholder:contentImageFloat},
                       fetches=[totalLoss, contentLoss, styleLoss], loss_callback=callback)

    final_image = session.run(initialImageVariable)

    # pixels are between 0 and 255
    final_image = final_image + np.array([128.0, 128.0, 128.0])
    final_image = np.clip(final_image, 0.0, 255.0)

    return final_image

initialImageVariable, styleImagePlaceholder, contentImagePlaceholder, contentLoss, styleLoss, totalLoss = CreateStyleGraph(vggData, contentLayers, styleLayers)

finalImage = Update(session, numberOfIterations, initialImageVariable, styleImagePlaceholder, contentImagePlaceholder, contentImageFloat, styleImageFloat, contentLoss, styleLoss, totalLoss)


#End the session
session.close()

#Revert image to viewable format
tempShape = finalImage.shape
finalImage = np.reshape(finalImage, tempShape[1:])
#Plot Image
plt.figure(4)
plt.imshow(finalImage/ 255.0, interpolation='sinc')

# Ensure the pixel-values are between 0 and 255.
finalImage = np.clip(finalImage, 0.0, 255.0)
# Convert to bytes.
finalImage = finalImage.astype(np.uint8)
# Write the image-file in jpeg-format.
with open(finalImageFileName, 'wb') as file:
    PIL.Image.fromarray(finalImage).save(file, 'jpeg')


