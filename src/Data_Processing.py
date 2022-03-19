#Importing necessary libraries and packages
import pandas as pd
from matplotlib import pyplot as plt


#Read data set from .csv file
df = pd.read_csv('DataSets/BTC_Price_USD.csv')


#Visualize data using plot / X axis is Date and Y axis is Price in USD
def visualiseDataInPlot():

    price = df[['Close']]

    plt.figure(figsize = (15,9))
    plt.plot(price)
    plt.xticks(range(0, df.shape[0], 50), df['Date'].loc[::50], rotation=45)
    plt.title('BTC price USD')
    plt.xlabel('Date')
    plt.ylabel('Price')

    plt.show()

#Preprocess Data
def removeNullValues():
    #Remove entire row from dataset if there are any null value
    df.dropna(inplace = True)