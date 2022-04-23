import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def calculateRMSE(actualTestData, predictedData, actualBTCPrice, predictedBTCPrice):

    # Evaluate model performance by using RMSE
    test_rmse = math.sqrt(mean_squared_error(actualTestData, predictedData))
    train_rmse = math.sqrt(mean_squared_error(actualBTCPrice, predictedBTCPrice))

    rmse = ['Train RMSE', 'Test RMSE']
    values = [train_rmse, test_rmse]

    #Plot RMSE
    plt.bar(rmse, values)
    plt.title('Train vs Test RMSE')
    plt.xlabel('RMSE')
    plt.ylabel('RMSE values')
    plt.show()