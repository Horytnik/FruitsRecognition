import pandas as pd
import os
from os.path import isfile
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten, Activation, BatchNormalization
from keras.utils import to_categorical
import cv2
import time
import matplotlib.pyplot as plt
import pickle

startTime = time.time()

photosMainPath = "D:/Studia_PJATK/FruitsRecognition/FruitsRecogintion/FruitsImages/"
photosPaths = []

print("Loading paths of the images...")
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

dfImagesPaths = pd.DataFrame(photosPaths, columns=["Fruit", "Path"])

# Amount of each fruit photos
plt.bar(dfImagesPaths.Fruit.value_counts().index,dfImagesPaths.Fruit.value_counts())
plt.xticks(rotation='vertical')
plt.show()

# Mapping fruit names to numbers
fruitNum = np.linspace(0, len(dfImagesPaths.Fruit.unique())-1,  len(dfImagesPaths.Fruit.unique()), dtype=int)
mappedNames = dict(zip(dfImagesPaths["Fruit"].unique(), fruitNum.T))
dfImagesPaths["Fruit"] = dfImagesPaths["Fruit"].map(mappedNames)
dfImagesPaths.Fruit = dfImagesPaths.Fruit.astype(int)

# Shuffle for splitting them
dfImagesPaths = shuffle(dfImagesPaths)
dfImagesPaths = dfImagesPaths.reset_index(drop=True)

# imgTrain, imgTest , labelTrain, labelTest = train_test_split(dfImagesPaths.Path, dfImagesPaths.Fruit, test_size=0.1)

imgTrain = dfImagesPaths.Path[0:int(0.2 * len(dfImagesPaths.Path))]
imgTest = dfImagesPaths.Path[int(-0.05 * len(dfImagesPaths.Path)):]

labelTrain = dfImagesPaths.Fruit[0:int(0.2 * len(dfImagesPaths.Fruit))]
labelTest = dfImagesPaths.Fruit[int(-0.05 * len(dfImagesPaths.Fruit)):]

labelTrain = to_categorical(labelTrain)


# CNN model creation
imgShape = (150,150,3)

print("Creating the model")
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

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# loading the images
fruitImages = []
fruitLabels = []

loadedImgCtr = 0
print("Loading images")
for idx, path in enumerate(imgTrain):
    loadedImage = plt.imread(path)
    loadedImage = cv2.resize(loadedImage,(150,150))

    loadedLabel = labelTrain[idx]

    fruitImages.append(loadedImage)
    fruitLabels.append(loadedLabel)
    loadedImgCtr += 1
    print("Loaded img {}".format(loadedImgCtr))

fruitImages = np.array(fruitImages)
fruitLabels = np.array(fruitLabels)


print("Training the model")

history = model.fit(fruitImages, fruitLabels, validation_split=0.1, epochs=100, batch_size=128)
# summarize history for accuracy
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.ylim(0.5, 1)
# plt.ylim(0, 1)
plt.show()

del fruitImages
del fruitLabels

# scores = model.evaluate(imgTest, labelTest, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))

filename = 'my_model.sav'
pickle.dump(model, open(filename, 'wb'))

endTime = time.time()
print("Processing images and training the model took {} seconds".format(endTime - startTime))