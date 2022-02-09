#Import necessary packages and library
import csv
import subprocess
import sys

#Download libraries if needed
subprocess.check_call([sys.executable, "-m", "pip", "install", "tweepy"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "csvFile"])
import tweepy

#Private key/secret codes for accessing Twitter
consumer_key= 'gRj2v2HHDMnBkCbXQCcvaUwID'
consumer_secret= 'PGYgR03BSZb0KC3A9zD4K5HkBiFewyXiHoi3vHoCItfLygQA7T'
access_token= '872378624682459136-zXD0nW4ZmPZv0EHeAxRQwzG93iFH8Ch'
access_token_secret= 'GA3bevXTwjAeRWz9LQFeWEoWHl2J95NIL2NbZLRZ3o1BN'

def authenticateUser():
    # Creating API object
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    return api

#API object
api = authenticateUser()

#Get Tweets from home page
def get_tweets_homepage():
    public_tweets = api.home_timeline()
    for tweet in public_tweets:
        print(tweet.text)



#Give name of twitter account by using "@"
#Example "@binance" or "@billgates"
#IMPORTANT:  All letter must be lowercase
#This function retrieves all tweets from a user and saves them into allTweetsFile.csv with important information about the tweet.
def get_tweets_from_account(accName):
    total_tweets_collected = 0
    #Open or create if doesn't exist allTweetsFile.csv
    csvFile = open('allTweetsFile.csv', 'a')
    csvWriter = csv.writer(csvFile)

    #Retrieve maximum amount of tweets from timeline of account
    tweets = tweepy.Cursor(api.user_timeline, screen_name=accName).items()

    #loop through all tweets and extract date, Id and text of the tweet and save it into the .csv file
    for tweet in tweets:
        csvWriter.writerow([tweet.created_at, tweet.id, tweet.text.encode('utf-8')])


#Get tweets from timeline of the following users
get_tweets_from_account("@coindeskmarkets")
get_tweets_from_account("@coindesk")
get_tweets_from_account("@btctn")
get_tweets_from_account("@btc_archive")
get_tweets_from_account("@bitcoin")
get_tweets_from_account("@wublockchain")
get_tweets_from_account("@blockworks_")