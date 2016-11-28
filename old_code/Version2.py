# PSTAT 131 Project
# Jason Freeberg
# Fall Quarter 2016

# Modules
import string, re, sys, time
import pymongo as mongo
from helpers import *  # my user defined functions
from sklearn import naive_bayes, metrics
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from numpy import array, append

# Main function
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
    N_COMMON_WORDS = 100
    FOLDS = 3
    # Regular Expressions
    regex = re.compile('[^%s]+' % re.escape(string.punctuation))
    digits = re.compile('[^%s]+' % re.escape(string.digits))

    # https://regex101.com/r/FCVdpt/1

    # Point to the MongoDB collection
    client = mongo.MongoClient(host="localhost", port=27017)
    collection = client[DATABASE_NAME][COLLECTION_NAME]

    # Pull parts of every JSON. Drop _id field
    print("Query")
    mongoPull = collection.find({}, {"overall": 1,
                                     "reviewerID": 1,
                                     "unixReviewTime": 1,
                                     "reviewText": 1,
                                     "_id": 0}).limit(10000)

    reviews = array(list(mongoPull))

    # Train / test split
    kfold = KFold(n=len(reviews), n_folds=FOLDS, shuffle=True, random_state=seed)

    all_precision = array([])
    all_recall = array([])

    print("Train/Test Split")
    for train_index, test_index in kfold:
        train, test = reviews[train_index], reviews[test_index]

        vectorizer = TfidfVectorizer(stop_words="english")

        trainLabels = list()
        trainText = list()
        testLabels = list()
        testText = list()

        for json in train:
            trainLabels.append(json["overall"])
            trainText.append(json["reviewText"])
        for json in test:
            testLabels.append(json["overall"])
            testText.append(json["reviewText"])

        traintfidf = vectorizer.fit_transform(trainText)
        testtfidf = vectorizer.transform(testText)

        print("Train Dim =", traintfidf.shape)
        print("Train label=", len(trainLabels))
        print("Test Dim =", testtfidf.shape)
        print("Test label=", len(testLabels))

        # Train and test naive bayes classifier
        print("Modeling")
        # class_prior=[400, 300, 550, 1400, 4000]
        classifier = naive_bayes.MultinomialNB(class_prior=[400, 300, 550, 1400, 4000])
        classifier.fit(X=traintfidf, y=trainLabels)
        print("class count=", classifier.class_count_)
        #predictions = classifier.predict(testtfidf)
        probs = classifier.predict_proba(testtfidf)
        predictions = sequential_comparison(probs)
        print("Class probabilities:", probs[:15])

        print("Validation")
        print(metrics.confusion_matrix(y_true=testLabels, y_pred=predictions))
        precision = metrics.precision_score(y_true=testLabels, y_pred=predictions, average="macro")
        recall = metrics.recall_score(y_true=testLabels, y_pred=predictions, average="macro")
        all_precision = append(all_precision, precision)
        all_recall = append(all_recall, recall)

        print("Precision =", round(precision, 4))
        print("Recall  =", round(recall, 4))
        print("\n ------------------------ \n")

    mean_precision = all_precision.mean()
    mean_recall = all_recall.mean()

    print("Mean precision =", round(mean_precision, 3))
    print("Mean recall =", round(mean_recall, 3))

    end = time.time()
    print("Entire process took", round(end - start, 2), "seconds.\n... Goodbye.")
