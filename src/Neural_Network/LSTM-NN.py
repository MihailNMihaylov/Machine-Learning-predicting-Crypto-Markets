#Import necessary packages and libraries
import numpy as np
import pandas as pd
import subprocess
import sys
from src.Data_Processing import *
#subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

dataset = pd.read_csv('../../src/DataSets/CombinedResults.csv')
price = dataset[['Close']]

#Visualize data using plot / X axis is Date and Y axis is Price in USD
def visualizeDataset():

    plt.figure(figsize = (15,9))
    plt.plot(price)
    plt.xticks(range(0, dataset.shape[0], 50), dataset['Date'].loc[::50], rotation=45)
    plt.title('BTC price USD')
    plt.xlabel('Date')
    plt.ylabel('Price')

    plt.show()


processedDf = preprocessDataset(dataset)


#Split Dataset
predictionDays = 60
train_Df = processedDf[:len(processedDf) - predictionDays].values.reshape(-1,1)
test_Df = processedDf[len(processedDf) - predictionDays:].values.reshape(-1,1)


#Rescale Dataset
train_Scaled = MinMaxScaler(feature_range=(0,10)).fit_transform(train_Df)
test_Scaled = MinMaxScaler(feature_range=(0,10)).fit_transform(test_Df)

#ref https://github.com/rohan-paul/MachineLearning-DeepLearning-Code-for-my-YouTube-Channel/blob/master/Finance_Stock_Crypto_Trading/Bitcoin_Price_Prediction_with_LSTM.ipynb
def dataset_prep_lstm(df, look_back=5):

    dataX, dataY = [], []

    for i in range(len(df) - look_back):
        window_size_x = df[i:(i + look_back), 0]
        dataX.append(window_size_x)
        dataY.append(df[i + look_back, 0])  # this is the label or actual y-value
    return np.array(dataX), np.array(dataY)


trainX, trainY = dataset_prep_lstm(train_Scaled)
testX, testY = dataset_prep_lstm(test_Scaled)

#Reshape and prepare datasets from lstm model
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1 ))


model = Sequential()

model.add(LSTM(units=128, activation='relu', retuen_sequence = True, input_shape = (trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=64, input_shape= (trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units=1))

print(model.summary())
