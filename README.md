# Fruits Recognition
Recognition of fruits in live camera based on photos

# Results
The aim of the project was to create an algorithm which can recognize fruits from live camera view. The result is shown in below example of four fruits. 

<img src = "https://github.com/Horytnik/FruitsRecognition/blob/master/FruitRecognitionWorkingExample.gif" />

In the project I was basing on photos dataset found on https://www.kaggle.com/chrisfilo/fruit-recognition and my own photos of fruits which I did by PC camera and script. 
The distribution of fruits photos is shown below: 

<img src = "https://github.com/Horytnik/FruitsRecognition/blob/master/readmeImages/fruitsDistribution.jpg" />

Final model accuracy is shown below:

<img src = "https://github.com/Horytnik/FruitsRecognition/blob/master/readmeImages/modelAccuracy.jpg" />

# Description
In the project were used different types of photos, like groups of fruits, different landscapes and also no fruit detection. At the beginning photos were sorted into groups and augmented to have more samples which gaves around 150k of photos. Randomly selected 30k was used to train the model. Used model is sequential CNN model with kernel size [3,3] and relu as activation function. 

# Main parts od the model
* Conv2D - convolutional layer. It performs convolution which is a linear operation that involves the multiplication of a set of weights with the input.
* MaxPooling2D - it is an operation which took maximum value from selected are which allows to downsample the picture.
* Flatten - It transforms a two-dimensional matrix of features into a vector which can be connected to clasifier.
* Dense - layer which connects input with output.
* Dropout - randomly removes the desired amount of connections between neurons.

# Files explanation

* fruitDetectionModel.py - this file contains several parts: finding photos in desired directory, photos distribution graph, creation and training of model, saving the created model.
* detectFruit.py - this file contains the video capture from OpenCV and detection by loaded model
* my_model.sav - this is the best acheved model
* photoCapture.py - this is script which was used to capture the photos of fruits
* photosAugmentation.py - this file was use to augument photos. This allowes to create around 150k photos.
