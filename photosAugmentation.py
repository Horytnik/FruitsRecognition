import os
from numpy import expand_dims
from keras.preprocessing.image import load_img, save_img
from keras.preprocessing.image import ImageDataGenerator


# pathToAugment = "D:/Studia_PJATK/FruitsRecognition/FruitsRecogintion/MyTestImages/MyCreatedPhotos/AfterAugmentation/"
pathToAugment = "D:/Studia_PJATK/FruitsRecognition/FruitsRecogintion/MyTestImages/MyCreatedPhotos/tempAugmentation/"
images = []


def photoAug(mainPath, dataGen, type, photoAmount=3):
    samples = expand_dims(loadedImg, 0)

    it = dataGen.flow(samples, batch_size=1)
    for i in range(0, photoAmount):
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        img_name = mainPath + folder + "/" + file[:-4] + "_aug" + type + "_{}.png".format(
            i)
        save_img(img_name, image)

for folder in os.listdir(pathToAugment):
    print(folder + "...")
    for file in os.listdir(pathToAugment+folder):
        images.append(load_img(pathToAugment+folder + "/" + file))
        loadedImg = load_img(pathToAugment+folder + "/" + file)

        datagen = ImageDataGenerator(width_shift_range=[-100, 100])
        photoAug(pathToAugment, datagen, "Wid")

        datagen = ImageDataGenerator(height_shift_range=[-100, 100])
        photoAug(pathToAugment, datagen, "Hei")

        datagen = ImageDataGenerator(rotation_range=90)
        photoAug(pathToAugment, datagen, "Rot", 6)

        datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
        photoAug(pathToAugment, datagen, "Zoom", 6)

print("end")

