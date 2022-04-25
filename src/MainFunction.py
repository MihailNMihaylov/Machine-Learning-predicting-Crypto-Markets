#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Run this file to compile the system
#In the getTweets() function on line 26 change the date to current date
#Running this file is going to take more than 1 hour !!
#Files needed for Neural Network are already presented in the system directory
#Run file LSTM_No_Sentiment.py to run the Neural Network model 1
#Run file LSTM_Sentiment.py to run the Neural Network model 2
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#This is the main file where everything is going to start
from src.Data_preprocessing import *
from src.Twitter_API.TwitterScraping import *
from src.BTC_Stats_Retriever.Bitcoin_Market_download import *
from src.Sentiment_Analysis_Tweets.TextBlobSentimentAnalysis import *
from src.Sentiment_Analysis_Tweets.VaderSentimentAnalysis import *
from datetime import datetime

#Automatically downloading Bitcoin market statistics
def get_BTC_Dataset():
    downloadBitcoinStats()

#Automatically retrieve tweets from accounts and save them into Tweets_Data_Scrapping.csv
def get_Tweets():

    listOfUsersnames = ["coindeskmarkets", "coindesk", "BTCTN", "btc_archive", "bitcoin", "wublockchain", "blockworks_"]
    file = None
    for username in listOfUsersnames:
        print(username)
        #Change the date below to current date in the same format!!!
        file = Get_tweets_by_user(username, "2017-04-25")

    file.close()
    print('File closed!')

#This function evaluates the ability of Textblob and Vader to detect positive text
def evaluateSentimentAnalyzers():

    #Text blob analyzer on sample text
    sentimentResultTextBlob("This is a sample textcreated for evaluation of both Textblob  and Vader library for sentiment analysis!! "
        "This message has mainly positive news to determine whether which one of the analyzers will perofrm better"
        "Bitcoin is now accepted as a payment method in most supermarkets across the UK"
        "China government approvs Bitcoin as an official currency"
        "Crypto currencies and especially Bitcoin are widely used by world banks"
        "Bitcoin tend to increase its volume double in size in the next two years "
                            "Investing in crypto will make you happy and wealthy"
                            "Trading is now more reliable, joyful and enjoyable with the new features introduced by major trade markets"
                            "Good news and bright future is ahead of the crypto world as many millionaires choose to invest their forture in crypto"
                            "as it has the potential to double their investment in shoe time ")

    #Vader analyzer on sample text
    sentimentResultVader("This is a sample textcreated for evaluation of both Textblob  and Vader library for sentiment analysis!! "
        "This message has mainly positive news to determine whether which one of the analyzers will perofrm better"
        "Bitcoin is now accepted as a payment method in most supermarkets across the UK"
        "China government approvs Bitcoin as an official currency"
        "Crypto currencies and especially Bitcoin are widely used by world banks"
        "Bitcoin tend to increase its volume double in size in the next two years "
                            "Investing in crypto will make you happy and wealthy"
                            "Trading is now more reliable, joyful and enjoyable with the new features introduced by major trade markets"
                            "Good news and bright future is ahead of the crypto world as many millionaires choose to invest their forture in crypto"
                            "as it has the potential to double their investment in shoe time ")

#Delete all files except 5DaysActualPrice before calling Run() function!!
def Run():
    evaluateSentimentAnalyzers()
    get_BTC_Dataset()
    get_Tweets()
    CalculateSentimentScoreOfTweets()
    combineSentimentAndBTCStats()

#Function that loads and runs everything
#Calling function Run() below, is going to take more than 1 hour to download all tweets and perform sentiment analysis on them
#Files are already downloaded in the directory
#Run file LSTM_No_Sentiment.py to run the Neural Network model 1
#Run file LSTM_Sentiment.py to run the Neural Network model 2

#Run()