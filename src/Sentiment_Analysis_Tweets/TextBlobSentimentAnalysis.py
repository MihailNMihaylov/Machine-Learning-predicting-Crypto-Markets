#Importing packages and libraries
import subprocess
import sys

#Installing textblob library if missing
subprocess.check_call([sys.executable, "-m", "pip", "install", "textblob"])

#Importing TextBlob
from textblob import TextBlob


#Function for sentiment analysis using TextBlob
def sentimentResultTextBlob(sentence):

    sentimentScore = TextBlob(sentence).sentiment
    print("Compound sentiment score of Textblob: {:.5f}".format(sentimentScore[0]))


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Testing the function (CODE BELOW IS NOT PART OF THE IMPLEMENTATION)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#Sample tweets to test the accuracy of the sentiment analysis of TextBlob
#sentimentResultTextBlob("This is a sample textcreated for evaluation of both Textblob  and Vader library for sentiment analysis!! "
                       # "This message has mainly positive news to determine whether which one of the analyzers will perofrm better"
                        #"Bitcoin is now accepted as a payment method in most supermarkets across the UK"
                        #"China government approvs Bitcoin as an official currency"
                        #"Crypto currencies and especially Bitcoin are widely used by world banks"
                        #"Bitcoin tend to increase its volume double in size in the next two years ")