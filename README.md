# Fruits Recognition
Recognition of fruits based on photos

# Introduction
The aim of the project was to create an algorithm which can recognize fruits from live camera view. The result is shown in below example of four fruits. 

<img src = "https://github.com/Horytnik/FruitsRecognition/blob/master/FruitRecognitionWorkingExample.gif" />

In the project I was basing on photos dataset found on https://www.kaggle.com/chrisfilo/fruit-recognition and my own photos of fruits which I did by PC camera and script. 
The distribution of fruits photos is shown below: 

<img src = "https://github.com/Horytnik/FruitsRecognition/blob/master/readmeImages/fruitsDistribution.jpg" />

Final model accuracy is shown below:

<img src = "https://github.com/Horytnik/FruitsRecognition/blob/master/readmeImages/modelAccuracy.jpg" />

# Files explanation

* fruitDetectionModel.py - this file contains several parts: finding photos in desired directory, photos distribution graph, creation and training of model, saving the created model.
* detectFruit.py - this file contains the video capture from OpenCV and detection by loaded model
* my_model.sav - this is the best acheved model
* photoCapture.py - this is script which was used to capture the photos of fruits
* photosAugmentation.py - this file was use to augument photos. This allowes to create around 150k photos.
