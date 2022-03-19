#This is the main file where everything is going to start
from src.Data_Processing import *
from src.Twitter_API.TwitterScraping import *


#Preprocessing of BTC dataset
def preprocess_BTC_Dataset():
    removeNullValues()
    visualiseDataInPlot()

#Automatically retrieve tweets from accounts and save them into Tweets_Data_Scrapping.csv
def get_Tweets():
    listOfUsersnames = ["coindeskmarkets", "coindesk", "BTCTN", "btc_archive", "bitcoin", "wublockchain", "blockworks_"]
    file = None
    for username in listOfUsersnames:
        print(username)
        file = Get_tweets_by_user(username, "2022-03-01")
    file.close()
    print('File closed!')

get_Tweets()

