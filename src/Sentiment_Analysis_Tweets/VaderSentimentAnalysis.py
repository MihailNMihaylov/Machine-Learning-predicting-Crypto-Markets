#Importing packages and libraries
import subprocess
import sys

#Installing textblob library if missing
subprocess.check_call([sys.executable, "-m", "pip", "install", "vaderSentiment"])

#Importing Vader and TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


#Function for sentiment analysis using Vader
def sentimentResultVader(sentenceToAnalyze):

    vaderObj = SentimentIntensityAnalyzer()

    sentimentScore = vaderObj.polarity_scores(sentenceToAnalyze)

    return sentimentScore['compound']



#Sample tweets to test the accuracy of the sentiment analysis of Vader
print(sentimentResultVader("Employees are asking to be #PaidInBitcoin, and we listened. With our new Bitcoin Savings Plan, companies can now offer an innovative compensation perk for employees to securely buy, sell, and hold #Bitcoin without incurring any transaction or storage fees. https://bit.ly/3IRnv8g"))
print(sentimentResultVader("This is the worst day"))