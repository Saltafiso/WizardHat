from functools import partial


import numpy as np
from sklearn import *
import pickle

class Classifier:
    """Base class for classifier."""

    # create a classifier object
    def __init__(self, classifier=None, filename=None):
        self.classifier = None
        self.changeModel(classifier, filename)
    # predict result of data
    def predict(self, data):
        return self.classifier.predict(data)
    
    def predict_proba(self, data):
        return self.classifier.predict_proba(data)
    # train the classifier with sample data and coresponding features
    def train(self, samples, features):
        self.classifier.fit(samples, features)
    # save the model to a file
    def saveModel(self,filename):
        pickle.dump(self.classifier, open(filename, 'wb'))
    # input classifier method: load a new classifier to the object
    # input filename : load the classifier from file to the object
    def changeModel(self, classifier=None, filename=None):
        if filename:
            self.classifier = pickle.load(open(filename, 'rb'))
        if classifier:
            self.classifier = classifier
        else:
            raise "Must provide a method or a model"
