import pickle
import functions
import pandas as pd
import cv2
import numpy as np

modelFilename = 'finalized_model.sav'
loaded_model = pickle.load(open(modelFilename, 'rb'))
fruitNames = ['OrgFileName','Apple', 'Banana', 'Carambola', 'Guava', 'Kiwi', 'Mango', 'Orange', 'Peach', 'Pear', 'Persimmon', 'Pitaya', 'Plum', 'Pomegranate', 'Tomatoes', 'muskmelon']

testPath = "D:/Studia_PJATK/FruitsRecognition/FruitsRecogintion/MyTestImages/ToTest/"

images, fruitFilenames = functions.load_all_images(testPath)

prediction = loaded_model.predict(images)

dfPred = pd.concat( [pd.DataFrame(fruitFilenames),pd.DataFrame(prediction)], axis = 1)
dfPred.columns = fruitNames

print('End part1')

cap = cv2.VideoCapture(0)

while (True):
    X = []
    ret, img = cap.read()
    # x, y, w, h = cv2.boundingRect()
    image = cv2.rectangle(img, (20,20), (50, 50), (36, 255, 12), 1)
    image = cv2.resize(image, (150, 150))
    X.append(image)
    imgToPred = np.array(X)
    vidPrediction = loaded_model.predict(imgToPred)
    dfVidPrediction = pd.DataFrame(vidPrediction)
    dfVidPrediction.columns = fruitNames[1:]
    predName = dfVidPrediction.idxmax(axis = 1)
    predVal = dfVidPrediction.max(axis = 1) * 100
    cv2.putText(image, predName[0], (50, 50 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 150, 12), 2)
    cv2.putText(image, str(predVal[0]), (30, 30 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 150, 12), 2)
    cv2.imshow('img', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



print('end')