import pickle
import functions
import pandas as pd
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


scaler = StandardScaler()
minMaxScaler = MinMaxScaler()

dfPredTable = pd.DataFrame()

# modelFilename = 'finalized_model.sav'

modelFilename = 'my_model3.sav'
fruitNames = ['OrgFileName','Apple', 'Banana', 'Carambola', 'Guava', 'Kiwi', 'Mango', 'NoFruit', 'Orange', 'Peach', 'Pear', 'Persimmon', 'Pitaya', 'Plum', 'Pomegranate', 'Tomato', 'muskmelon']

loaded_model = pickle.load(open(modelFilename, 'rb'))

testPath = "D:/Studia_PJATK/FruitsRecognition/FruitsRecogintion/MyTestImages/ToTest/"

images, fruitFilenames = functions.load_all_images(testPath)

prediction = loaded_model.predict(images)

dfPred = pd.concat( [pd.DataFrame(fruitFilenames),pd.DataFrame(prediction)], axis = 1)
dfPred.columns = fruitNames

print('End part 1')

cap = cv2.VideoCapture(0)

while (True):
    X = []
    ret, imgCam = cap.read()

    im_rgb = cv2.cvtColor(imgCam, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(im_rgb, (150, 150), interpolation=cv2.INTER_CUBIC)
    image = frame / 255

    X.append(image)
    imgToPred = np.array(X)
    vidPrediction = loaded_model.predict(imgToPred)
    dfVidPrediction = pd.DataFrame(vidPrediction)
    dfVidPrediction.columns = fruitNames[1:]

    # Calculate mean of last x photos
    if(dfPredTable.__len__() < 50):
        dfPredTable = dfPredTable.append(dfVidPrediction, ignore_index=True)
    else:
        dfPredTable = dfPredTable.drop(0)
        dfPredTable = dfPredTable.append(dfVidPrediction, ignore_index=True)
    columnsMean = dfPredTable.mean(axis=0)

    predName = columnsMean.idxmax(axis = 1)
    predVal = columnsMean.max() * 100
    # print(predVal)
    cv2.putText(imgCam, predName, (50, 50 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 150, 12), 2)
    cv2.putText(imgCam, str(predVal.round(2)), (30, 30 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 150, 12), 2)

    cv2.imshow('CameraView', imgCam)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


print('end')