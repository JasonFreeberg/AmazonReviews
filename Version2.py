# Jason Freeberg
# November 2016
# Git: /JasonFreeberg

# Updated pipeline.

"""
Step 1: Query MongoDB
Step 2: Create k-Fold cross validation iterator

FOR train and test sets in cross validation iterator DO:
    Step 3: Using training data...
        a: Aggregate features from text
        b: Create feature space
            - Top N words
            - Month text was written
            - Character count
    Step 4: Map train and test observations onto feature space
    Step 5: Classify test set observations
        a: Create table of predicted and actual classes
    Step 6: Save validation results
        a: Append to a table outside of loop
    END

Step 7: Examine results
"""

import string, re, sys, time
import pymongo as mongo
from sklearn.cross_validation import KFold
from numpy import array, append
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import nltk

def match(a_list):
    correct = 0
    total = len(a_list)
    for item in a_list:
        if item[1] == item[0]:
            correct += 1

    return correct, total


def tokenize(string, lower=True):
    """
    Returns the string tokenized as words. If lower=True, the tokens are lowercased
    """
    tokens = word_tokenize(string)
    if lower:
        tokens = [token.lower() for token in tokens]
    return tokens


def filter_tokens(list_of_tokens, regexpr, remove_stopwords=True):
    """
    Takes in a list of tokens, returns list of tokens that match regex
    """
    return [token
            for token in list_of_tokens
            if regexpr.match(token) and
            token not in stopwords.words("english")
            if remove_stopwords]


def clean(string, regexpr, lower = True, remove_stopwords=True):
    tokens = word_tokenize(string)
    filtered_tokens = [token
                       for token in tokens
                       if regexpr.match(token) and
                       token not in stopwords.words("english")
                       if remove_stopwords]
    if lower:
        return [token.lower() for token in filtered_tokens]
    else:
        return filtered_tokens

def checkprint(i, m, string):
    if i % m == 0:
        print(string, i)
    return i + 1


def feature_space(n, list_of_lists):
        token_set = []
        i = 0
        for list_of_tokens in list_of_lists:
            i = checkprint(i, 20, "Extracting")
            token_set += list_of_tokens
        return set(Tuple[0] for Tuple in FreqDist(token_set).most_common(n))


def feature_space2(n, list_of_strings, regexpr, remove_stopwords=True):
    token_set = []
    i = 0
    for string in list_of_strings:
        i = checkprint(i, 20, "Extracting")
        token_set += clean(string, regexpr, remove_stopwords=remove_stopwords)
    return set(Tuple[0] for Tuple in FreqDist(token_set).most_common(n))


def transform(feature_space, list_of_lists, list_of_ratings):
        assert len(list_of_lists) == len(list_of_ratings)
        i = 0
        transformed = []
        for list_of_tokens, rating in zip(list_of_lists, list_of_ratings):
            i = checkprint(i, 20, "Transforming")
            return_dict = {}
            return_dict["charcount"] = len(list_of_tokens)

            for feature in feature_space:
                return_dict[feature] = (feature in list_of_tokens)


            transformed.append((return_dict, rating))

        return transformed


def to_tuple(list_of_dict):
        return [(Tuple[0], Tuple[1]) for Tuple in list_of_dict]

# Main Function
if __name__ == '__main__':
    start = time.time()

    # DB Setup
    DB_NAME = "amazon"
    COLLECTION_NAME = "electronics"

    FOLDS = 2
    N_COMMON_WORDS = 75
    SEED = 131

    # Regular expressions
    punct = re.compile('[^%s]+' % re.escape(string.punctuation))

    client = mongo.MongoClient(host="localhost", port=27017)
    collection = client[DB_NAME][COLLECTION_NAME]

    # Pull parts of every JSON. Drop _id field
    print("Start query")
    query = collection.find({}, {"overall": 1,
                                 "reviewText": 1,
                                 "_id": 0}).limit(600)

    reviews = array([])
    ratings = array([])
    i = 0
    for json in query:
        reviews = append(reviews, clean(json["reviewText"], punct))
        ratings = append(ratings, json["overall"])
        i = checkprint(i, 50, "Cleaned")
    print("End query.")

    k_fold = KFold(n=len(reviews), n_folds=FOLDS, random_state=SEED)

    all_predictions = []
    for train_index, test_index in k_fold:
        print("Split.")

        train_text, test_text = reviews[train_index], reviews[test_index]
        train_labels, test_labels = ratings[train_index], ratings[test_index]

        start_features = time.time()

        # simple_features = feature_space(N_COMMON_WORDS, train, "reviewText", punct)
        simple_features = feature_space2(N_COMMON_WORDS, train_text, punct)
        trainFeatureLabels = transform(simple_features, train_text, train_labels)
        testFeatureLabels = transform(simple_features, test_text, test_labels)

        stop_features = time.time()

        feature_time = round((stop_features - start_features)/60, 2)
        print("Creating feature space and transforming took", feature_time, "minutes.")

        start_modeling = time.time()

        classifier = nltk.NaiveBayesClassifier.train(trainFeatureLabels)
        predictions = [(classifier.classify(Tuple[0]), Tuple[1]) for Tuple in testFeatureLabels]

        stop_modeling = time.time()
        model_time = round((stop_modeling - start_modeling)/60, 2)
        print("Training and testing model took", model_time)

        all_predictions += predictions

    correct, total = match(all_predictions)
    print("# of tested observations =", total)
    print("# correct =", correct)
    print("% correct =",  round(correct/total, 2))


    stop = time.time()
    total_time = round((stop - start) / 60, 2)
    print("Took", total_time, "minutes to cross validate 6000 observations.")