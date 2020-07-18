import pickle
import functions
import pandas as pd

modelFilename = 'finalized_model.sav'
loaded_model = pickle.load(open(modelFilename, 'rb'))
fruitNames = ['OrgFileName','Apple', 'Banana', 'Carambola', 'Guava', 'Kiwi', 'Mango', 'Orange', 'Peach', 'Pear', 'Persimmon', 'Pitaya', 'Plum', 'Pomegranate', 'Tomatoes', 'muskmelon']

testPath = "D:/Studia_PJATK/FruitsRecognition/FruitsRecogintion/MyTestImages/ToTest/"

images, fruitFilenames = functions.load_all_images(testPath)

prediction = loaded_model.predict(images)

dfPred = pd.concat( [pd.DataFrame(fruitFilenames),pd.DataFrame(prediction)], axis = 1)
dfPred.columns = fruitNames

print('end')