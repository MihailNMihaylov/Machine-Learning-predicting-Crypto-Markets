#Importing necessary libraries and packages
import csv
from src.Sentiment_Analysis_Tweets.VaderSentimentAnalysis import *
import pandas as pd
from matplotlib import pyplot as plt
from dateutil.parser import parse
from collections import OrderedDict
import numpy as np

#Read data set from .csv file
#df = pd.read_csv('DataSets/BTC_Price_USD.csv')


#Visualize data using plot / X axis is Date and Y axis is Price in USD
def visualiseBitcoinDataInPlot():

    #price = df[['Close']]

    #plt.figure(figsize = (15,9))
    #plt.plot(price)
    #plt.xticks(range(0, df.shape[0], 50), df['Date'].loc[::50], rotation=45)
    #plt.title('BTC price USD')
    #plt.xlabel('Date')
    #lt.ylabel('Price')

    plt.show()

#Preprocess Data
#def removeNullValuesBitcoin():
    #df.dropna(inplace = True)

#Method for calculating the sentiment score of each tweet
#Saves the result into SemtimentResult.csv file
#Each day has a sentiment score for all the tweets on that day
def CalculateSentimentScoreOfTweets():

    #Empty dictonary that holds the sentiment results for each day
    dictSentimentScore = {}

    #Open the tweets file and convert it into Numpy Array
    data = pd.read_csv('../src/DataSets/Tweets_Data_Scraping.csv')
    array = data.to_numpy()

    #Iterate over each row
    for row in array:

        #Perform sentiment analysis on each tweet
        sentimentCompoundScore = sentimentResultVader(row[1])

        if row[0] in dictSentimentScore.keys():
            dictSentimentScore[row[0]].append(sentimentCompoundScore)
        else:
            dictSentimentScore[row[0]] = []
            dictSentimentScore[row[0]].append(sentimentCompoundScore)

    #Calculate average sentiment score for each day
    for row in dictSentimentScore:
        avgScore = sum(dictSentimentScore[row]) / len(dictSentimentScore[row])
        dictSentimentScore[row] = avgScore

    #Order by date
    orderedDictSentimentScore = OrderedDict(sorted(dictSentimentScore.items()))

    #Save result into SentimentResult.csv file
    with open('../src/DataSets/SentimentResult.csv', 'a') as f:
        f.write('Date,SentimentScore\n')
        for key in orderedDictSentimentScore.keys():
            f.write("%s,%s\n" % (key, "{:.4f}".format(float(orderedDictSentimentScore[key]))))
        f.close()

def combineSentimentAndBTCStats():
    btc_stats = pd.read_csv('../src/DataSets/DownloadBTC-USD.csv')
    twitterSentimentResult = pd.read_csv('../src/DataSets/SentimentResult.csv')

    combinedDF = pd.concat([btc_stats, twitterSentimentResult['SentimentScore']], axis=1, join='inner')
    combinedDF.to_csv("../src/DataSets/CombinedResults.csv", index=False)