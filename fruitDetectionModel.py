import pandas as pd
import os
from os.path import isfile
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten, Activation, BatchNormalization
import cv2

import matplotlib.pyplot as plt

photosMainPath = "D:/Studia_PJATK/FruitsRecognition/FruitsRecogintion/FruitsImages/"
photosPaths = []

# collecting paths of fruit images
for folder in os.listdir(photosMainPath):
    print(folder + "...")
    for item in os.listdir(photosMainPath+folder):
        if isfile(photosMainPath+folder + "/" + item):
            photosPaths.append([folder, photosMainPath+folder + "/" + item])
        else:
            print(item + "...")
            for photo in os.listdir(photosMainPath+folder + "/" + item):
                photosPaths.append([folder, photosMainPath + folder + "/" + item + "/" + photo])

print("Collecting image paths finished")

dfImagesPaths = pd.DataFrame(photosPaths, columns=["Fruit", "Path"])

# Amount of each fruit photos
plt.bar(dfImagesPaths.Fruit.value_counts().index,dfImagesPaths.Fruit.value_counts())
plt.xticks(rotation='vertical')
plt.show()

# Mapping fruit names to numbers
fruitNum = np.linspace(0, len(dfImagesPaths.Fruit.unique()),  len(dfImagesPaths.Fruit.unique()), dtype=int)
mappedNames = dict(zip(dfImagesPaths["Fruit"].unique(), fruitNum.T))
dfImagesPaths["Fruit"] = dfImagesPaths["Fruit"].map(mappedNames)
dfImagesPaths.Fruit = dfImagesPaths.Fruit.astype(int)

# Shuffle for splitting them
dfImagesPaths = shuffle(dfImagesPaths)

imgTrain, imgTest , fruitTrain, fruitTest = train_test_split(test_size=0.2)

# CNN model creation
imgShape = (150,150,3)

model = Sequential()

# CNN layer with 64 filters, each one have 3x3 size. Pandding same means that we don't change size of the image
model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=imgShape, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=imgShape, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=imgShape, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), input_shape=imgShape, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.2)) # randomly removes 20% of existing neurons connections to prevent overfitting

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(len(mappedNames))) # last layer with amount of neurons the same as amount of fruits
model.add(Activation('softmax')) # Selects the neuron with highest probability

epochs = 25

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# loading the images

for i, path in enumerate(dfImagesPaths.Path):
    image = plt.imread(path)
    image = cv2.resize(image,(150,150))