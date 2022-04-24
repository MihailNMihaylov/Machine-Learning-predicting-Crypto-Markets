#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Run this file to compile the Neural Network model 2 (using sentiment score)
#Run this file AFTER running the MainFunction file that gathers the necessary dataset!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#Importing necessary packages and libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout

from src.Neural_Network.CalculateRMSE import calculateRMSE
from src.Neural_Network.PredictFuture5Days import predictFutureDays

#Load dataset
dataset = pd.read_csv("../../src/DataSets/CombinedResults.csv")

def visualizeData():
    # Plot BTC Price
    plt.figure(figsize=(15, 9))
    plt.plot(dataset['Close'])
    plt.xticks(range(0, dataset.shape[0], 50), dataset['Date'].loc[::50], rotation=45)
    plt.title('BTC price USD')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


#Visualize BTC price data and daily sentiment score
visualizeData()

groupedDataset = dataset[['Close','SentimentScore']].groupby(dataset['Date']).mean()

price = groupedDataset['Close'].values.reshape(-1,1)
sentiment = groupedDataset['SentimentScore'].values.reshape(-1,1)

values = price.astype('float32')
sentiment = sentiment.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

train_size = int(len(scaled) * 0.8)
test_size = len(scaled) - train_size
train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]

split = train_size

def convert_dataset_to_ML(dataset, look_back, sentiment, sent=False):
    dataX, dataY = [], []

    for i in range(len(dataset) - look_back):
        if i >= look_back:
            a = dataset[i-look_back:i+1, 0]
            a = a.tolist()
            if(sent==True):
                a.append(sentiment[i].tolist()[0])
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
    #print(len(dataY))
    return np.array(dataX), np.array(dataY)

look_back = 7
trainX, trainY = convert_dataset_to_ML(train, look_back, sentiment[0:train_size],sent=True)
testX, testY = convert_dataset_to_ML(test, look_back, sentiment[train_size:len(scaled)], sent=True)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(units=128, activation='relu', input_shape = (trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64, input_shape= (trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(loss='mean_squared_error', optimizer='adam')


history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), verbose=1, shuffle=False)


results = model.evaluate(testX, testY, batch_size=100)

predictedTestData = model.predict(testX)
predictedTrainData = model.predict(trainX)

predictedTestData = scaler.inverse_transform(predictedTestData.reshape(-1, 1))
predictedTrainData = scaler.inverse_transform(predictedTrainData.reshape(-1, 1))

trainY = scaler.inverse_transform(trainY.reshape((-1, 1)))
testY = scaler.inverse_transform(testY.reshape((-1, 1)))




#Plot Results
#Plot loss function of train and test dataset
plt.figure(figsize=(16,7))
plt.title("Loss function of train and test dataset")
plt.plot(history.history['loss'], label= 'Train Loss')
plt.plot(history.history['val_loss'], label= 'Test Loss')
plt.legend()
plt.show()

#Plot actual vs predicted dataset
plt.figure(figsize=(16,7))
plt.title("Actual data vs predicted data")
plt.plot(predictedTestData,'r', marker='.',  label= 'Predicted data')
plt.plot(testY, marker='.', label= 'Actual data')
plt.legend()
plt.show()

#Evaluate model performance by using RMSE
calculateRMSE(testY, predictedTestData, trainY, predictedTrainData)

#Predict Future 5 days
predictFutureDays(model, testX, scaler, predictedTestData)