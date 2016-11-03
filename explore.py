# PSTAT 131 Project
# Jason Freeberg
# Fall Quarter 2016

# Extract features from the review text

# Modules
import string, re, sys
import pymongo as mongo
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.api import StringTokenizer
from nltk.corpus import stopwords
from collections import Counter

# UDFs
def filter_tokens(list_of_tokens, regexpr):
    """
    Takes in a list of tokens, returns list of tokens that match regex
    """
    return [word for word in list_of_tokens if regexpr.match(word)]

def high_rating(review_rating):
    """
    Reduce overall ratings to binary: 1 if review is over 3 stars, else 0
    """
    if review_rating >= 3:
        return 1
    else:
        return 0

# Main function
if __name__ == '__main__':

    """
    # Read the file's parameters
    with open(sys.argv[1], 'r') as f:
        args = f.readlines()

    args = [item.strip('\n') for item in args]

    # Parse the database, collection, and location
    DATABASE_NAME = args[0]
    COLLECTION_NAME = args[1]
    """

    # Database and Collection names
    DATABASE_NAME = "amazon"
    COLLECTION_NAME = "electronics"

    # Training proportion for naive bayes classifier
    TRAIN_PROPORTION = 0.85

    # Point to the MongoDB collection
    client = mongo.MongoClient(host="localhost", port=27017)
    collection = client[DATABASE_NAME][COLLECTION_NAME]

    # Pull parts of every JSON. Drop _id field
    mongoPull = collection.find({}, {"overall": 1,
                                        "reviewerID": 1,
                                        "unixReviewTime": 1,
                                        "reviewText": 1,
                                        "_id": 0}).limit(100)
    reviews = (list(mongoPull))

    # Matches punctuation
    regex = re.compile('[^%s]+' % re.escape(string.punctuation))

    # Iterate Extract features
    for dict in reviews:
        review = dict["reviewText"]
        sentences = sent_tokenize(review)  # list of strings, one element is one sentence
        token_sent = [word_tokenize(sentence) for sentence in sentences] # List of lists of tokenized sentences

        dict["three_or_greater"] = high_rating(dict["overall"])
        dict["charcount"] = len(review)
        dict["words"] = word_tokenize(review)
        dict["filter_token_sent"] = [filter_tokens(sublist, regex) for sublist in token_sent]

        #runningvar, object
    for i, review in enumerate(reviews):
        print("RAW TEXT:", review["reviewText"])
        print("TOKENIZED WORDS:", review["words"])
        print("TOKENIZED BY SENT.:", review["filter_token_sent"])
        print("------------------------------------------------------------------------")
        if i == 2:
            break

    # Train / test split
    numReviews = len(reviews)
    trainsize = np.ceil(numReviews * TRAIN_PROPORTION)
    train, test = reviews[:trainsize], reviews[trainsize:]

