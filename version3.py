# PSTAT 131 Project
# Jason Freeberg
# Fall Quarter 2016

# Modules
import string, re, sys, time
import pymongo as mongo
from helpers import *  # my user defined functions
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.multiclass import OneVsRestClassifier
from numpy import array, random


if __name__ == '__main__':
    seed = 1
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
    FOLDS = 2

    # Regular Expressions
    # https://regex101.com/r/FCVdpt/1
    regex = re.compile('[^%s]+' % re.escape(string.punctuation))
    digits = re.compile('[^%s]+' % re.escape(string.digits))

    # Point to the MongoDB collection
    client = mongo.MongoClient(host="localhost", port=27017)
    collection = client[DATABASE_NAME][COLLECTION_NAME]

    print("-- Querying --")

    query = collection.find({}, {"overall": 1,
                                 "reviewText": 1,
                                 "_id": 0}).limit(10000)

    print("-- Splitting Data --")

    data = array(list(query))
    dev_data, test_data = train_test_split(data, test_size=.15)

    # Separate reviews and ratings
    reviews = [json["reviewText"] for json in dev_data]
    labels = [json["overall"] for json in dev_data]

    print("-- Model Tuning --")

    # Pipeline components
    vectorizer = TfidfVectorizer(stop_words="english")
    clusterer = KMeans(max_iter=200)
    classifier1 = SVC(kernel="linear", random_state=seed, cache_size=500, class_weight="balanced", probability=True)
    classifier2 = MultinomialNB()
    mutliclassifier1 = OneVsRestClassifier(classifier1)
    mutliclassifier2 = OneVsRestClassifier(classifier2)

    # Assemble pipeline
    pipeline1 = Pipeline([
        ('vectorizer', vectorizer),
        ('clusterer', clusterer),
        ('classifier', mutliclassifier1),
    ])

    pipeline2 = Pipeline([
        ("vectorizer", vectorizer),
        ('clusterer', clusterer),
        ("classifier", mutliclassifier2),
    ])

    # Hyper-parameters to search over
    #ngram_range = [(1, 1), (1, 2)]
    ngram_range = [(1, 2)]

    n_clusters = [2, 5, 8]

    #C = [1, 2, 4, 8, 10]
    C = [2]
    kernel = ["linear"]

    priors = [sorted([random.exponential(scale=10) for i in range(0, 5)], reverse=True) for i in range(0, 1)]
    #alpha = random.uniform(1.5, 3, 1)
    alpha = [2.47131]

    parameters1 = {
        "vectorizer__ngram_range": ngram_range,
        "clusterer__n_clusters": n_clusters,
        "classifier__estimator__C": C
    }

    parameters2 = {
        "vectorizer__ngram_range": ngram_range,
        "classifier__estimator__alpha": alpha,
        # "classifier__estimator__class_prior": priors,
        "clusterer__n_clusters": n_clusters
    }

    # Tune the pipeline
    tuned_model1 = GridSearchCV(pipeline1, parameters1, scoring="f1_micro", cv=FOLDS, verbose=10)
    tuned_model2 = GridSearchCV(pipeline2, parameters2, scoring="f1_micro", cv=FOLDS, verbose=10)

    print("Tuning first model...")
    tuned_model1.fit(X=reviews, y=labels)

    print("Tuning second model...")
    tuned_model2.fit(X=reviews, y=labels)

    print("-- Model Validation --")

    # Move this to be with the other split
    test_reviews = [json["reviewText"] for json in test_data]
    test_labels = [json["overall"] for json in test_data]

    # Get predicted labels and probabilities
    predictions1 = tuned_model1.predict(test_reviews)
    predictions2 = tuned_model2.predict(test_reviews)
    prob_pred1 = tuned_model1.predict_proba(test_reviews)
    prob_pred2 = tuned_model2.predict_proba(test_reviews)

    # Model validation metrics
    conf_matrix1 = metrics.confusion_matrix(test_labels, predictions1)
    conf_matrix2 = metrics.confusion_matrix(test_labels, predictions2)
    precision1 = metrics.precision_score(y_true=test_labels, y_pred=predictions1, average="macro")
    precision2 = metrics.precision_score(y_true=test_labels, y_pred=predictions2, average="macro")
    recall1 = metrics.recall_score(y_true=test_labels, y_pred=predictions1, average="macro")
    recall2 = metrics.recall_score(y_true=test_labels, y_pred=predictions2, average="macro")

    # Print summary output
    print(len(dev_data), "reviews in dev data.")
    print(len(test_data), "reviews in test data.")

    print("Model 1...")
    print("Best params: \n", tuned_model1.best_params_)
    print("Precision is {:0.3f} and recall is {:1.3f}.".format(precision1, recall1))
    print(conf_matrix1)

    print("Model 2...")
    print("Best params: \n", tuned_model2.best_params_)
    print("Precision is {:0.3f} and recall is {:1.3f}.".format(precision2, recall2))
    print(conf_matrix2)
