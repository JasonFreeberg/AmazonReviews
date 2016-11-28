# Amazon consumer review data project
# Jason Freeberg
# Helper function definitions used in the project

from nltk import FreqDist
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn import linear_model
from operator import ge


def filter_tokens(list_of_tokens, regexpr):
    """
    Takes in a list of tokens, returns list of tokens that match regex
    """
    return [word for word in list_of_tokens if not regexpr.search(word)]


def get_word_set(list_of_dict, key, regexpr, remove_stopwords = True):
        all_words = []

        for dict in list_of_dict:
            words = [word for word in dict[key].split()]
            words = [word.lower() for word in words]
            if regexpr:
                words = filter_tokens(words, regexpr)
            if remove_stopwords:
                words = [word for word in words if word not in stopwords.words("english")]

            all_words += words

        return FreqDist(all_words)


def make_features(list_of_tokens, common_words):
        """
        Takes a list of tokens and set of N common words
        Returns a dictionary with N key/value pairs. Indicating the presence of that common word in the list of tokens
        """
        return_list = list()
        return_list.append(len(list_of_tokens))
        for common_word in common_words:
            return_list.append(common_word in list_of_tokens)

        return return_list


def match(list1, list2):
        """
        len(list1) == len(list2)
        :return: Returns a tuple of the number of matching elements, and a logical list of equal length
        """
        n_matches = 0
        matches = []
        if len(list1) != len(list2):
            raise ArithmeticError
        else:
            for index, element1 in enumerate(list1):
                if element1 == list2[index]:
                    n_matches += 1
                    matches.append(1)
                else:
                    matches.append(0)

        return n_matches, matches


def high_rating(review_rating):
    """
    Reduce overall ratings to binary: 1 if review is over 3 stars, else 0
    """
    if review_rating == 5:
        return 1
    else:
        return 0


class SlidingThreshold:
    def __init__(self, thresholds=None):
        self.thresholds = thresholds
        self.X = None

    def transform(self):
        return_list = list()

        for List in self.X:
            logical = list(map(ge, List, self.thresholds))
            pred = logical.index(True) + 1.0
            return_list.append(pred)

        return return_list

    def fit(self, X):
        self.X = X


def sliding_threshold(list_of_lists, thresholds):
    """
    Takes a list of lists holding predicted class probabilities, and returns a list of class predictions using
    the thresholds given.
    """
    return_list = list()
    for List in list_of_lists:
        logical = list(map(ge, List, thresholds))
        pred = logical.index(True) + 1.0
        return_list.append(pred)
    return return_list


def sequential_neighbor(list_of_probs):
    """
    Assumes that the class probabilities are ordered by increasing frequency in the data
    ... example_probs = [least_frequent_label_prob, moderately_frequent_label_prob, ..., most_frequent_label_prob]

    predicted_label := least_frequent_label_prob
    For element in list[1:]:
        if predicted_label > element:
            return predicted_label's *index*
        else:
            predicted_label := element
            continue through list

    """
    list_of_probs = list(list_of_probs)
    pred_prob = list_of_probs[0]
    next_prob = list_of_probs[1]
    i = 2
    while pred_prob < next_prob:
        i += 1
        pred_prob = next_prob
        try:
            next_prob = list_of_probs[i]
        except IndexError:
            continue
    return list_of_probs.index(pred_prob) + 1.0

def sequential_comparison(list_of_lists):
    """
    Applies sequential_neighbor over a list of lists of probabilities.
    """
    return [sequential_neighbor(List) for List in list_of_lists]

"""
a = [.2, .1, .21, .13, .22]
b = [.1, .2, .3, .4, .5]
r = [a, b]

print(sequential_neighbor(a))
print(sequential_neighbor(b))
print(sequential_comparison(r))
"""