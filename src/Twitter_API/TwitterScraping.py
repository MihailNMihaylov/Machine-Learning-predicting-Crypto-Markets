import datetime
import subprocess
import  sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "snscrape"])

import snscrape.modules.twitter as sntwitter
import csv


#Set number for maximum tweets
maxTweets = 45000

#Open (or create if doesn't exist) "Tweets_Data_Scraping.csv" file to save retrieved tweets by Id, Date and Text
csvFile = open('../src/DataSets/Tweets_Data_Scraping.csv', 'a', newline='', encoding='utf8')

csvWriter = csv.writer(csvFile)
csvWriter.writerow([ 'Date', 'Tweet'])

#Get tweets from User since a specific date
#write username as the first parameter without "@"
#Write date in format "yyyy-mm-dd"
def Get_tweets_by_user(username, sinceDate):

    #Variable that counts how many tweets are there in a day
    sameDayTweets = 0

    #Initial date converted from DateTime to use only "yyyy-mm-dd" (without the hour)
    firstDate = datetime.datetime.now()
    firstDate = datetime.datetime(firstDate.year, firstDate.month, firstDate.day)

    #Import search query into Twitter search bar and start retrieving and saving tweets into csv file
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:@{} + since:{}'.format(username, sinceDate)).get_items()):

        #Convert current tweet date to be only "yyyy-mm-dd" (without the hour)
        tweetDate = datetime.datetime(tweet.date.year, tweet.date.month, tweet.date.day)

        if firstDate != tweetDate:
            sameDayTweets = 0

        #Check if there are more than 3 tweets in a day from that user
        if firstDate == tweetDate:
            sameDayTweets = sameDayTweets + 1

            #Max tweets per day - 2
            if sameDayTweets > 1:
                continue

        #if max number of tweets is reached break the loop and stop retrieving
        if i > maxTweets:
            break

        #Keep content of tweets on the same line
        tweet.content = tweet.content.replace('\n', ', ')

        csvWriter.writerow([tweetDate, tweet.content])
        firstDate = tweetDate
    return csvFile







#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Testing the function (CODE BELOW IS NOT PART OF THE IMPLEMENTATION)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#Call the function for all users that have influence on the price or have relevant information about the price

#Get_tweets_by_user("coindesk", "2014-01-01")
#Get_tweets_by_user("coindeskmarkets", "2018-01-01")
#Get_tweets_by_user("BTCTN", "2015-08-01")
#Get_tweets_by_user("btc_archive", "2018-04-01")
#Get_tweets_by_user("bitcoin", "2014-01-01")
#Get_tweets_by_user("wublockchain", "2014-01-01")
#Get_tweets_by_user("blockworks_", "2018-05-01")
