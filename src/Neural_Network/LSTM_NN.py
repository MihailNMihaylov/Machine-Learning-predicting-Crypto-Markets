#Import necessary packages and libraries
import numpy as np
import subprocess
import sys
from src.Data_preprocessing import preprocessDataset, MinMaxScaler, pd
#subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.models import load_model

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
train_Scaler = MinMaxScaler(feature_range=(0, 1))
train_Scaled = train_Scaler.fit_transform(train_Df)
test_Scaler = MinMaxScaler(feature_range=(0, 1))
test_Scaled = test_Scaler.fit_transform(test_Df)

#ref https://github.com/rohan-paul/MachineLearning-D)eepLearning-Code-for-my-YouTube-Channel/blob/master/Finance_Stock_Crypto_Trading/Bitcoin_Price_Prediction_with_LSTM.ipynb
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

#Create/ Configure network
model = Sequential()

model.add(LSTM(units=128, activation='relu', input_shape = (trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64, input_shape= (trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units=1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

savedModel = "model.hdf5"

#Configuration settings of the model
checkpoint = ModelCheckpoint(filepath=savedModel, monitor='loss', verbose=1, save_best_only=True, mode='min')

#Fit/Compile the model
history = model.fit(trainX, trainY, batch_size=32, epochs=100, verbose=1, shuffle=False, validation_data=(testX, testY), callbacks= checkpoint)

#Plot loss of training and testing dataset
plt.figure(figsize=(16,7))
plt.title("Loss of train and test dataset")
plt.plot(history.history['loss'], label= 'train')
plt.plot(history.history['val_loss'], label= 'test')
plt.legend()
plt.show()

#Make a prediction on the test dataset
predictedData = model.predict(testX)
predictedData = test_Scaler.inverse_transform(predictedData.reshape(-1, 1))

actualTestData = test_Scaler.inverse_transform(testY.reshape((-1, 1)))


plt.figure(figsize=(16,7))
plt.title('Predicted vs Actual Test data')
plt.plot(predictedData, 'r', marker='.', label='Predicted Data')
plt.plot(actualTestData, marker='.', label = 'Actual Data')
plt.legend()
plt.show()

#Make a prediction by using the saved model
loadModel = load_model(savedModel)

predictedBTCPrice = loadModel.predict(trainX)
predictedBTCPrice = train_Scaler.inverse_transform(predictedBTCPrice.reshape(-1, 1))
actualBTCPrice = train_Scaler.inverse_transform(trainY.reshape(-1, 1))

plt.figure(figsize=(16,7))
plt.title("Predicted vs Actual Train data")
plt.plot(predictedBTCPrice, 'r', marker='.', label='Predicted Train Data')
plt.plot(actualBTCPrice, marker='.', label = 'Actual Train Data')
plt.legend()
plt.show()


#Evaluate model performance by using RMSE
test_rmse = math.sqrt(mean_squared_error(actualTestData, predictedData))
print('Test RMSE: %.3f' % test_rmse)

train_rmse = math.sqrt(mean_squared_error(actualBTCPrice, predictedBTCPrice))

print('Train RMSE: %.3f' % train_rmse)


#Predict future 5 days BTC price
daysPeriod = 5
testX_last_days = testX[testX.shape[0] - daysPeriod:]

predicted_future_days = []

for i in range(5):
    predicted = loadModel.predict(testX_last_days[i:i+1])
    predicted = test_Scaler.inverse_transform(predicted.reshape(-1,1))

    predicted_future_days.append(predicted)

predicted_future_days = np.array(predicted_future_days)
predicted_future_days = predicted_future_days.flatten()
predictedData = predictedData.flatten()

concat_predictions = np.concatenate((predictedData, predicted_future_days))

plt.figure(figsize=(16,7))
plt.title("Predicted future 5 days")
plt.plot(concat_predictions, 'r', marker='.', label='Predicted 5 days')
plt.plot(predictedData, marker='.', label = 'Actual Train Data')
plt.legend()
plt.show()