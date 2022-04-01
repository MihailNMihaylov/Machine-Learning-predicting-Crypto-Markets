import math
from sklearn.metrics import mean_squared_error

def calculateRMSE(actualTestData, predictedData, actualBTCPrice, predictedBTCPrice):
    # Evaluate model performance by using RMSE
    test_rmse = math.sqrt(mean_squared_error(actualTestData, predictedData))
    print('Test RMSE: %.3f' % test_rmse)

    train_rmse = math.sqrt(mean_squared_error(actualBTCPrice, predictedBTCPrice))

    print('Train RMSE: %.3f' % train_rmse)