import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM

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

    # Plot Sentiment Score
    plt.figure(figsize=(15, 9))
    plt.plot(dataset['SentimentScore'])
    plt.xticks(range(0, dataset.shape[0], 50), dataset['Date'].loc[::50], rotation=45)
    plt.title('Sentiment Score')
    plt.xlabel('Date')
    plt.ylabel('SentimentS Score')
    plt.show()

    # Plot BTC price and Sentiment Score combined to see any correlation
    fig, axs = plt.subplots(2)
    fig.suptitle('BTC price compared to Sentiment Score')
    axs[0].plot(dataset['Date'][1400:], dataset['Close'][1400:])
    axs[0].set_title('BTC price in USD')
    axs[1].plot(dataset['Date'][1400:], dataset['SentimentScore'][1400:])
    axs[1].set_title('Sentiment Score')
    plt.show()

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

look_back = 5
trainX, trainY = convert_dataset_to_ML(train, look_back, sentiment[0:train_size],sent=True)
testX, testY = convert_dataset_to_ML(test, look_back, sentiment[train_size:len(scaled)], sent=True)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(100))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(trainX, trainY, epochs=200, batch_size=100, validation_data=(testX, testY), verbose=1, shuffle=False)


results = model.evaluate(testX, testY, batch_size=100)

predictedData = model.predict(testX)

plt.figure(figsize=(16,7))
plt.title("Loss of train and test dataset")
plt.plot(history.history['loss'], label= 'Train Loss')
plt.plot(history.history['val_loss'], label= 'Test Loss')
plt.legend()
plt.show()

plt.figure(figsize=(16,7))
plt.title("Actual vs Predicted dataset")
plt.plot(predictedData, label= 'Actual')
plt.plot(testY, label= 'Predicted')
plt.legend()
plt.show()