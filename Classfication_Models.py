from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn import utils
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier
import time
import os
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

class Classfication():
    def __init__(self, x_train,y_train, x_test):
        self.X = x_train
        self.y = y_train
        self.X_test = x_test
    def SVC(self, kernel='linear'):
        '''
        :param X: Input data
        :param y: Actual output
        :param X_test: Test data
        :param kernel: kernel used initially linear
        :return: model used and classified data
        '''
        classification = SVC()

        if kernel == 'rbf':
            classification = SVC(kernel='rbf')
            return  self.run_model(classification, str(classification))
        elif kernel == 'poly':
            classification = SVC(kernel='poly')
            return  self.run_model(classification, str(classification))
        else:
            classification = SVC(kernel='linear')
            return  self.run_model(classification, str(classification))


    def decicionTreeClassifier(self):
        classification = DecisionTreeClassifier()
        return self.run_model(classification, str(classification))

    def SGDClassifier(self):
        classification= SGDClassifier()
        return self.run_model(classification, str(classification))


    def KNN(self, k):
        classification = KNeighborsClassifier(n_neighbors=k)
        return self.run_model(classification, str(classification))

    def Random_forest_classifier(self):
        classification = RandomForestClassifier(n_estimators=100, max_depth=40)
        return self.run_model(classification, str(classification))

    def adaboast_classiefier(self):
        ## 4  0.1   150
        classification = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),learning_rate=0.2, algorithm="SAMME",  n_estimators=80)
        return self.run_model(classification, str(classification))

    def Extractor_Classsifier(self):
        classification = ExtraTreesClassifier(n_estimators=180)
        return self.run_model(classification, str(classification))

    def Gradient_boost_Classsifier(self):
        ## 150 3
        classification = GradientBoostingClassifier(n_estimators=150, max_depth=3)
        return self.run_model(classification, str(classification))


    def gaussian_naive_bayesian_classifier(self):
        classification = GaussianNB()
        return self.run_model(classification, str(classification))

    def multilayer_preceptron(self, hidden_layers=(100,)):
        classification = MLPClassifier(hidden_layer_sizes=hidden_layers)
        return self.run_model(classification, str(classification))

    def logistic_regression(self, X, y, X_test):
        classification = LogisticRegression()
        return self.run_model(classification, str(classification))


    def run_model(self, model, model_name ):
        train_time = 0
        if (os.path.exists("Saved_Models/Classification/" + model_name.split('(')[0] + ".pkl")):
            model = pickle.load(open("Saved_Models/Classification/"+ model_name.split('(')[0] +".pkl", 'rb'))
        else:
            start_time = time.time()
            model= model.fit(self.X, self.y)
            train_time = round((time.time() - start_time), 4)
            pickle.dump(model, open("Saved_Models/Classification/" + model_name.split('(')[0] + ".pkl", 'wb'))
        start_time = time.time()
        y_predict = []
        if not len(self.X_test) == 0:
           y_predict = model.predict(self.X_test)
        test_time = round((time.time() - start_time), 4)
        return y_predict, train_time, test_time


    def cv_parameter_tuning(self, kf_splits, x_train, y_train):
        print('###################### Cross-validation => Parameter Tuning ######################')

        max_dep = [3, 10, 20, 40]
        n_estim = [10, 50, 100]

        dt_max_dep_scores = []
        for i in max_dep:
            dt_model = RandomForestClassifier(n_estimators=50, max_depth=i)       #m20 + n50
            scores = (cross_val_score(dt_model, x_train, y_train, cv=kf_splits, scoring='accuracy')).mean()
            dt_max_dep_scores.append(scores)
        print(dt_max_dep_scores)

        dt_n_estim_scores = []
        for i in n_estim :
            dt_model = RandomForestClassifier(n_estimators=i, max_depth=20)       #m20 +n100
            scores = (cross_val_score(dt_model, x_train, y_train, cv=kf_splits, scoring='accuracy')).mean()
            dt_n_estim_scores.append(scores)
        print(dt_n_estim_scores)

