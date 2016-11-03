# Amazon consumer review data project
# Jason Freeberg
# Helper function definitions used in the project

from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

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


def get_word_set(list_of_dict, key, regexpr, remove_stopwords = True):
        all_words = []

        for dict in list_of_dict:
            words = word_tokenize(dict[key])
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
        return_dict = {}
        for common_word in common_words:
            return_dict[common_word] = (common_word in list_of_tokens)

        return return_dict