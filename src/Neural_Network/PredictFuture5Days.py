import numpy as np
import matplotlib.pyplot as plt

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

    plt.figure(figsize=(16, 7))
    plt.title("Predicted future 5 days")
    plt.plot(concat_predictions, 'r', marker='.', label='Predicted 5 days')
    plt.plot(predictedData, marker='.', label='Actual Train Data')
    plt.legend()
    plt.show()