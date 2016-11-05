# PSTAT 131 Project
# Jason Freeberg
# Fall Quarter 2016

# Feasibility test.
# Makes predictions using only the set of N most common words in training set.
# Classes are made binary: rating > 3 OR rating <= 3
# Only a simple 2-fold train/test split.

# Modules
import string, re, sys, time
import pymongo as mongo
from helpers import *  # my user defined functions
from nltk.tokenize import word_tokenize
import nltk

# Main function
if __name__ == '__main__':
    start = time.time()

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
    N_COMMON_WORDS = 100

    # Matches punctuation
    regex = re.compile('[^%s]+' % re.escape(string.punctuation))

    # Point to the MongoDB collection
    client = mongo.MongoClient(host="localhost", port=27017)
    collection = client[DATABASE_NAME][COLLECTION_NAME]

    # Pull parts of every JSON. Drop _id field
    mongoPull = collection.find({}, {"overall": 1,
                                     "reviewerID": 1,
                                     "unixReviewTime": 1,
                                     "reviewText": 1,
                                     "_id": 0}).limit(6000)

    reviews = list(mongoPull)

    # Train / test split
    # NEED TO SHUFFLE THE OBSERVATIONS -> use random module
    numReviews = len(reviews)
    trainsize = int(numReviews * TRAIN_PROPORTION)
    train, test = reviews[:trainsize], reviews[trainsize:]

    # MODELING

    # Set of most common words from training set only.
    word_set = get_word_set(train, "reviewText", regex)
    common_words = set(Tuple[0] for Tuple in word_set.most_common(N_COMMON_WORDS))

    trainFeaturesLabels = []
    testFeaturesLabels = []

    # Iterate over train and test sets, extract features
    for json in train:
        review = json["reviewText"]
        words = word_tokenize(review)
        words_lower = [word.lower() for word in words]

        features = make_features(words_lower, common_words)
        label = high_rating(json["overall"])

        # For now just use the dict
        trainFeaturesLabels.append((features, label))

    for json in test:
        review = json["reviewText"]
        words = word_tokenize(review)
        words_lower = [word.lower() for word in words]

        features = make_features(words_lower, common_words)
        label = high_rating(json["overall"])

        testFeaturesLabels.append((features, label))

    # Separate lists of test set's features and labels
    testFeatures = [tuple[0] for tuple in testFeaturesLabels]
    testLabels = [tuple[1] for tuple in testFeaturesLabels]

    # Train and test naive bayes classifier
    classifier = nltk.NaiveBayesClassifier.train(trainFeaturesLabels)
    predictions = [classifier.classify(features) for features in testFeatures]
    Ncorrect, matches = match(predictions, testLabels)

    # OUTPUT
    print("Cardinality of common word set =", len(common_words))
    print("Train size =", len(train))
    print("Test size =", len(test))
    print("# tested =", len(matches))
    print("# correct =", Ncorrect)
    print("% correct =", round(Ncorrect / len(matches), 4))

    if len(reviews) != len(train) + len(test):
        raise Warning("Train and test set sizes sum correctly.")
    if len(test) != len(matches):
        raise Warning

    end = time.time()
    print("Entire process took", round(end - start, 2), "seconds.\n... Goodbye.")