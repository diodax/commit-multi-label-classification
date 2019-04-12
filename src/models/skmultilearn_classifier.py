import numpy as np

from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.adapt import MLkNN
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


class ScikitMultilearnClassifier(object):
    """
    The class that contains the methods related to the Scikit-multilearn classifier.
    """

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.x_test = x_test

        self.y_train = y_train
        self.y_test = y_test

    def apply_binary_relevance(self):
        classifier = BinaryRelevance(LogisticRegression(solver='lbfgs'))
        classifier.fit(self.x_train, self.y_train)
        predictions = classifier.predict(self.x_test)
        a = accuracy_score(self.y_test, predictions)
        f = f1_score(self.y_test, predictions, average='macro')
        return a, f

    def apply_classifier_chain(self):
        classifier = ClassifierChain(LogisticRegression(solver='lbfgs'))
        classifier.fit(self.x_train, self.y_train)
        predictions = classifier.predict(self.x_test)
        a = accuracy_score(self.y_test, predictions)
        f = f1_score(self.y_test, predictions, average='macro')
        return a, f

    def apply_label_powerset(self):
        classifier = LabelPowerset(LogisticRegression(solver='lbfgs', multi_class='auto'))
        classifier.fit(self.x_train, self.y_train)
        predictions = classifier.predict(self.x_test)
        a = accuracy_score(self.y_test, predictions)
        f = f1_score(self.y_test, predictions, average='macro')
        return a, f

    def apply_mlknn(self):
        parameters = {'k': range(1, 5), 's': [0.5, 0.7, 1.0]}
        classifier = GridSearchCV(MLkNN(), parameters, scoring='f1_macro', cv=5, iid=False)
        # classifier = MLkNN(k=5)
        classifier.fit(self.x_train, self.y_train)
        # print(classifier.best_params_, classifier.best_score_)
        predictions = classifier.predict(self.x_test)
        a = accuracy_score(self.y_test, predictions)
        f = f1_score(self.y_test, predictions, average='macro')
        return a, f
