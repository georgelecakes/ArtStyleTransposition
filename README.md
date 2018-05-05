# ArtStyleTransposition
Project for 2018 Machine Learning Course at Rowan University using Convolutional Neural Networks to transfer style and content.

Project Setup:
This project was created and tested to run using Anaconda with the following packages:
-cudnn 6.0
-numpy 1.12.1
-pillow 5.0.0
-tensorflow-gpu 1.1.0

How to Run:
This program works best when run within an editor, such as Spyder. As such, it was implemented to be modified within the editor
and not via command line parameters. 

The following variables can be used to adjust the images used in processing, the output, number of iterations and loss Ratio:

Beginning on line 28 to 32:
contentImageName = r"images\Arthur.jpg"
styleImageName = r"images\starry-night.jpg"
finalImageFileName = r"Output.jpg"
numberOfIterations = 10000
lossRatio = 1e-3

If you would like to weight the different ReLU layers differently, line 72 can be modified:
styleLayerWeight = [0.2, 0.2, 0.2, 0.2, 0.2] #Divided amont the 5 relu levels, can adjust to change amount of influence

Once you have setup the programs to your liking, run it from your IDE or within a Python termainal.

References:
This project is based on several sources found across the internet.
Please refer to the following pages to find more reference material about style transfer.

Inceptionism: Going Deeper with Neural Networks
https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html

Artistic Style Transfer with Deep Neural Networks
https://shafeentejani.github.io/2016-12-27/style-transfer/

Artificially Artistic
https://github.com/cs60050/ArtificiallyArtistic/blob/master/README.md
