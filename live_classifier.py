import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize



class EnsembleClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


documents_f = open("pickled_algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features5k_f = open("pickled_algos/word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


def parse_text(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


# Load all classifiers from the pickled files

# function to load models given filepath
def load_model(file_path):
    classifier_f = open(file_path, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier


# Original Naive Bayes Classifier
ONB_Clf = load_model('pickled_algos/ONB_clf.pickle')

# Multinomial Naive Bayes Classifier
MNB_Clf = load_model('pickled_algos/MNB_clf.pickle')


# Bernoulli  Naive Bayes Classifier
BNB_Clf = load_model('pickled_algos/BNB_clf.pickle')

# Logistic Regression Classifier
LogReg_Clf = load_model('pickled_algos/LogReg_clf.pickle')

# Stochastic Gradient Descent Classifier
SGD_Clf = load_model('pickled_algos/SGD_clf.pickle')


ensemble_clf = EnsembleClassifier(ONB_Clf, MNB_Clf, BNB_Clf, LogReg_Clf, SGD_Clf)


def print_feats(d):
    z = []
    for key in d:
        if d[key]:
            z.append(key)
    return z


def sentiment(text):
    feats = parse_text(text)
    print(print_feats(feats))
    return ensemble_clf.classify(feats), ensemble_clf.confidence(feats)


text_a = 'This movie was so bad'

# print(sentiment(text_a))

text_a = '''The problem is with the corporate anticulture that controls these productions-and 
            the fandom-targeted demagogy that they're made to fulfill-which responsible casting 
                can't overcome alone.'''
text_b = '''Does it work? The short answer is: yes. There's enough to keep both diehard 
                Marvel fans and newcomers engaged.'''
text_c = '''It was lacking, a bit flat, and I'm honestly concerned about how she will enter
            the Marvel Cinematic Universe...it's so concerned with being a feminist film that 
            it forgets how to be a superhero movie.'''
text_d = '''The film may be about women breaking their shackles, but the lead actress feels kept 
            in check for much of the picture. Humor winds up being provided by Samuel Jackson's Nick 
            Fury, heart by Lashana Lynch's Maria Rambeau, and pathos by...well, it ain't Larson'''
text_e = '''"Everything was beautiful and nothing hurt"'''

print('a. ',sentiment(text_a), '\nb. ',
      sentiment(text_b), '\nc. ',
      sentiment(text_c), '\nd. ',
      sentiment(text_d), '\ne. ',
      sentiment(text_e))
