#This is the main file where everything is going to start
from src.Data_preprocessing import *
from src.Twitter_API.TwitterScraping import *
from src.BTC_Stats_Retriever.Bitcoin_Market_download import *
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
        file = Get_tweets_by_user(username, "2017-03-28")

    file.close()
    print('File closed!')


#get_BTC_Dataset()
#get_Tweets()
#CalculateSentimentScoreOfTweets()
combineSentimentAndBTCStats()