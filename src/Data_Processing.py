#Importing necessary libraries and packages
import csv
from src.Sentiment_Analysis_Tweets.VaderSentimentAnalysis import *
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#Read data set from .csv file
df = pd.read_csv('DataSets/BTC_Price_USD.csv')


#Visualize data using plot / X axis is Date and Y axis is Price in USD
def visualiseBitcoinDataInPlot():

    price = df[['Close']]

    plt.figure(figsize = (15,9))
    plt.plot(price)
    plt.xticks(range(0, df.shape[0], 50), df['Date'].loc[::50], rotation=45)
    plt.title('BTC price USD')
    plt.xlabel('Date')
    plt.ylabel('Price')

    plt.show()

#Preprocess Data
def removeNullValuesBitcoin():
    #Remove entire row from dataset if there are any null value
    df.dropna(inplace = True)

#Method for calculating and appending the sentiment compound score of each Tweet
#Updates DataFrame with Date, TweetContent, CompoundScore
def CalculateSentimentScoreOfTweets():

    #Open the tweets file
    data = pd.read_csv('../src/DataSets/Tweets_Data_Scraping.csv')
    array = data.to_numpy()

    #Iterate over each row
    for row in array:

        #Perform sentiment analysis on each tweet
        sentimentCompoundScore = sentimentResultVader(row[1])
        #Update the score
        row[2] = sentimentCompoundScore

    #Update the Tweets_Data_Scraping.csv file
    originalDF = open('../src/DataSets/Tweets_Data_Scraping.csv', 'r+')
    originalDF.truncate(0)
    originalDF.close()

    df = pd.DataFrame(array, columns=['Date', 'TweetContent', 'SentimentScore'])
    df.to_csv('../src/DataSets/Tweets_Data_Scraping.csv', index=False)




