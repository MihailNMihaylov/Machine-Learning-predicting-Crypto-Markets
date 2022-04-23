import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Syntax inspired by Rohan Paul, “MachineLearning-DeepLearning-Code-For-My-Youtube-Channel”, 2022, Github Repository: https://github.com/rohan-paul/MachineLearning-DeepLearning-Code-for-my-YouTube-Channel
def predictFutureDays(model, testX, test_Scaler, predictedData):

    # Predict future 5 days BTC price
    daysPeriod = 5
    testX_last_days = testX[testX.shape[0] - daysPeriod:]

    predicted_future_days = []

    for i in range(5):
        predicted = model.predict(testX_last_days[i:i + 1])
        predicted = test_Scaler.inverse_transform(predicted.reshape(-1, 1))

        predicted_future_days.append(predicted)

    predicted_future_days = np.array(predicted_future_days)
    predicted_future_days = predicted_future_days.flatten()
    predictedData = predictedData.flatten()

    concat_predictions = np.concatenate((predictedData, predicted_future_days))

    #Actual price for the predicted future 5 days has to be added manually into the csv file
    #format is Date,Close
    #Example: 2022-04-19,41502
    #Current values are relevant for the period of testing the system!
    fivedaysActualPrice = pd.read_csv("../../src/DataSets/5DaysActualPrice.csv")
    concat_actualFiveDays = np.concatenate((predictedData, fivedaysActualPrice['Close']))


    plt.figure(figsize=(16, 7))
    plt.title("Predicted future 5 days")
    plt.plot(concat_actualFiveDays, marker='.', label='Actual price for future 5 days')
    plt.plot(concat_predictions, 'r', marker='.', label='Predicted 5 days')
    plt.plot(predictedData,'b', marker='.', label='Actual Train Data')
    plt.legend()
    plt.show()