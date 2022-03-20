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

    print(sentimentScore)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Testing the function (CODE BELOW IS NOT PART OF THE IMPLEMENTATION)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#Sample tweets to test the accuracy of the sentiment analysis of TextBlob
#sentimentResultTextBlob("Employees are asking to be #PaidInBitcoin, and we listened. With our new Bitcoin Savings Plan, companies can now offer an innovative compensation perk for employees to securely buy, sell, and hold #Bitcoin without incurring any transaction or storage fees. https://bit.ly/3IRnv8g")