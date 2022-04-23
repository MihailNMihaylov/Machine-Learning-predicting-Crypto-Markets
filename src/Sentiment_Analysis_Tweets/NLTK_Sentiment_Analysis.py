#Install and Import libraries and packages necessary for text sentiment analysis

import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])

import nltk
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import re, string
import random
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


#Code partially inspired by https://realpython.com/python-nltk-sentiment-analysis/

#Packages and dataset for data normalization
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')

#This function removes noise and stop words from tweets
def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

#Converts list of tweets into dictionaries with key-tokens and value-True (Required for the model)
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


#Sample array of stop words
stop_words = stopwords.words('english')

#Convert tweets from .json format to a python list for further operation
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

text = twitter_samples.strings('tweets.20150430-223406.json')

#Split tweets text into tokens /single words
positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

#Cleans texts and saves them into two category lists - positive and negative
for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))


positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)


#Divide tweets into positive dataset and negative dataset
positive_dataset = [(tweet_dict, "Positive")
                     for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                     for tweet_dict in negative_tokens_for_model]

#Combining and shuffling the positive and negative tweets
dataset = positive_dataset + negative_dataset
random.shuffle(dataset)

#Splitting the dataset into 80% for training and 20% for testing
train_data = dataset[:8000]
test_data = dataset[8000:]

#Initialize and train model
classifier = NaiveBayesClassifier.train(train_data)

#Evaluate Model
print("Accuracy is:", classify.accuracy(classifier, test_data))



#Evaluate model on text that it has never seen before
custom_tweet = "This is a sample textcreated for evaluation of both NLTK package and Vader library for sentiment analysis!! " \
               "It consists of both positive and negative word. The amazing news are the fact that this way the test is going to be performed" \
               "Under the same conditions using the same text" \
               "This is amazing and price is expected to go up based on the new announcements, although it might seem awful to expected it to get any better" \
               "Similarly the chinese government now approved Bitcoin and It is now an official curreny widely used by many world banks which increases the volume" \
               "This might reflect into slow increase followed by a major decrease before the price remains stable."

custom_tokens = remove_noise(word_tokenize(custom_tweet))

print(classifier.classify(dict([token, True] for token in custom_tokens)))