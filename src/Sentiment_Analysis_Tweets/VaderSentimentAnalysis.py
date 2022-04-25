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

    print("Compound sentiment score of Vader: {:.5f}".format(sentimentScore['compound']))

    return sentimentScore['compound']


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Testing the function (CODE BELOW IS NOT PART OF THE IMPLEMENTATION)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#Sample tweets to test the accuracy of the sentiment analysis of Vader
#print("Compound sentiment score of Vader: {:.5f}".format(
#sentimentResultVader("This is a sample textcreated for evaluation of both Textblob  and Vader library for sentiment analysis!! "
#                        "This message has mainly positive news to determine whether which one of the analyzers will perofrm better"
#                        "Bitcoin is now accepted as a payment method in most supermarkets across the UK"
#                        "China government approvs Bitcoin as an official currency"
#                        "Crypto currencies and especially Bitcoin are widely used by world banks"
#                        "Bitcoin tend to increase its volume double in size in the next two years ")))
