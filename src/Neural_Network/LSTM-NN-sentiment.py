import pandas as pd
import numpy as np
from src.Data_preprocessing import preprocessDataset, MinMaxScaler

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from tensorflow.python.keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

datasetDF = pd.read_csv('../../src/DataSets/CombinedResults.csv')

processedDf = preprocessDataset(datasetDF, sentimentIncluded=True)
print(processedDf)
#Visualize BTC price and sentiment score
visualizeValuesDF = datasetDF.drop(['Open', 'High', 'Low', 'Dividends', 'Volume', 'Stock Splits'], axis='columns')
visualizeValuesDF.columns = ['Date', 'Close', 'SentimentScore']
values = visualizeValuesDF.values[:300]
groups = [ 1,2]
i = 1
plt.figure()
for group in groups:
	plt.subplot(len(groups), 1, i)
	plt.plot(values[:, group])
	plt.title(visualizeValuesDF.columns[group], y=0.5, loc='right')
	i += 1
plt.show()



value_scaler = MinMaxScaler(feature_range=(0,1))
value_scaled = value_scaler.fit_transform(processedDf.values)

def convert_values_for_NN(data, n_in=1, n_out=1, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

converted_values = convert_values_for_NN(value_scaled, 3, 1)
print(converted_values.head())

converted_values = converted_values.drop(converted_values.columns[-1], axis=1)
print(converted_values.head())

values = converted_values.values
n_train_hours = 170
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
print(train.shape)

train_X, train_y = train[:, :3], train[:, -1]
test_X, test_y = test[:, :3], test[:, -1]

train_X = train_X.reshape((train_X.shape[0], 3, 1))
test_X = test_X.reshape((test_X.shape[0], 3, 1))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

model = Sequential()
model.add(LSTM(128, activation='relu',  input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64, input_shape= (train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

#Save model file
savedModel = "modelSentiment.hdf5"
#Configuration settings of the model
checkpoint = ModelCheckpoint(filepath=savedModel, monitor='loss', verbose=1, save_best_only=True, mode='min')

# fit network
history = model.fit(train_X, train_y, epochs=150, batch_size=4, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], 3))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, -1:]), axis=1)
inv_yhat = value_scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, -1:]), axis=1)
inv_y = value_scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

plt.plot(inv_y)
plt.plot(inv_yhat)
plt.show()

inv_y = np.insert(inv_y,0,7000)
plt.plot(inv_y)
plt.plot(inv_yhat)
plt.show()