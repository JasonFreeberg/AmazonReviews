# PSTAT 131 Project
# Jason Freeberg
# Fall Quarter 2016

# Extract features from the review text

# Modules
import string, re, sys
import pymongo as mongo
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# UDFs
def clean_reviews(a_string, regexpr):
    """
    :param a_string: A string to be mapped
    :param regexpr: A regular expression to match
    :return: Returns a filtered list of lowercase words without punctuation or stop words
    """
    tokens = word_tokenize(a_string)
    noPunct = [regexpr.sub('', word) for word in tokens]
    return [word.lower() for word in noPunct if word not in stopwords.words("english") and word != '']

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

    DATABASE_NAME = "amazon"
    COLLECTION_NAME = "electronics"

    # Point to the MongoDB collection
    client = mongo.MongoClient(host="localhost", port=27017)
    collection = client[DATABASE_NAME][COLLECTION_NAME]

    # Pull the times, review text, and rating
    # drop _id field
    obs = collection.count()
    summaryStats = collection.find({}, {"overall": 1,
                                        "reviewerID": 1,
                                        "unixReviewTime": 1,
                                        "reviewText": 1,
                                        "_id": 0}).limit(100)

    reviewFrame = pd.DataFrame(list(summaryStats))
    print(reviewFrame.head())

    # Add column: character count of review
    reviewFrame["charcount"] = reviewFrame["reviewText"].map(lambda x: len(x))

    # Clean the review text and add column
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    reviewFrame["cleanText"] = reviewFrame.reviewText.map(lambda x: clean_reviews(x, regex))

    # Create dataframe of words and respective counts
    wordCounts = pd.DataFrame()
    flatReviews = pd.Series([word for sublist in reviewFrame["cleanText"] for word in sublist])
    counts = Counter(flatReviews)
    wordCounts["word"] = counts.keys()  # Condense down into constructor later
    wordCounts["count"] = counts.values()
    wordCounts.sort_values(by=["count"], ascending=False, inplace=True)
    print(wordCounts.head())
    top25 = wordCounts["word"][:25]

    # Create columns with frequency ratios of these top 25 words
    

    # Some output
    print("Mean Rating =", round(reviewFrame["overall"].mean(), 4))
    print("Mean number of characters =", round(reviewFrame["charcount"].mean(), 4))
    print("Top ten most common words =", list(top25))

