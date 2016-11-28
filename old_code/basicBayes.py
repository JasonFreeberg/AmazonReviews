# PSTAT 131 Project
# Jason Freeberg
# Fall Quarter 2016

# Modules
import string, re, sys, time
import pymongo as mongo
from helpers import *  # my user defined functions
from nltk.tokenize import word_tokenize
from sklearn import naive_bayes, metrics
from sklearn.cross_validation import KFold
from numpy import array


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

    all_precision = []
    all_recall = []

    for train_index, test_index in kfold:
        train, test = reviews[train_index], reviews[test_index]

        # MODELING
        # Set of most common words from training set only.
        print("Common Words")
        word_set = get_word_set(train, "reviewText", regex)
        common_words = set(Tuple[0] for Tuple in word_set.most_common(N_COMMON_WORDS))

        trainLabels = []
        trainFeatures = []
        testLabels = []
        testFeatures = []

        # Iterate over train and test sets, extract features
        print("Extract Features")
        for json in train:
            review = json["reviewText"]
            words = word_tokenize(review)
            words_lower = [word.lower() for word in words]

            features = make_features(words_lower, common_words)
            label = json["overall"]

            trainLabels.append(label)
            trainFeatures.append(features)

        for json in test:
            review = json["reviewText"]
            words = word_tokenize(review)
            words_lower = [word.lower() for word in words]

            features = make_features(words_lower, common_words)
            label = json["overall"]

            testLabels.append(label)
            testFeatures.append(features)

        # Train and test naive bayes classifier
        print("Modeling")
        classifier = naive_bayes.MultinomialNB()
        classifier.fit(X=trainFeatures, y=trainLabels)
        predictions = classifier.predict(testFeatures)

        print("Validation")
        precision = metrics.precision_score(y_true=testLabels, y_pred=predictions, average="macro")
        recall = metrics.recall_score(y_true=testLabels, y_pred=predictions, average="macro")
        all_precision.append(precision)
        all_recall.append(recall)

        print("Precision =", precision)
        print("Recall  =", recall)
        print("\n ------------------------ \n")

    mean_precision = sum(precision) / len(precision)
    mean_recall = sum(recall) / len(recall)

    print("Mean precision =", round(mean_precision, 3))
    print("Mean recall =", round(mean_recall, 3))

    end = time.time()
    print("Entire process took", round(end - start, 2), "seconds.\n... Goodbye.")
