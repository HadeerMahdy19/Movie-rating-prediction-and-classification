import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model, svm
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import time
import pickle
import os


class Regression:
    def __init__(self, x_train, y_train, x_test):
        self.X_train= x_train
        self.y_train = y_train
        self.x_test= x_test
    def Linear(self):
        '''
        :param X: Input data
        :param y: Actual output
        :param X_test: Test data
        :return: model used and Predicted data
        '''
        Linear_Model = linear_model.LinearRegression()
        return self.run_model(Linear_Model, str(Linear_Model), self.X_train, self.y_train, self.x_test)

    def Polynomial(self):
        '''
        :param X: Input data
        :param y: Actual output
        :param X_test: Test data
        :return: model used and Predicted data
        '''
        poly_model = linear_model.LinearRegression()
        polynomial_features = PolynomialFeatures(degree=2)
        x_train_poly = polynomial_features.fit_transform(self.X_train)
        x_test_poly = []
        if not  len(self.x_test) ==0:
            x_test_poly = polynomial_features.fit_transform(self.x_test)
        return self.run_model(poly_model, str('Polynomial_Regression'), x_train_poly, self.y_train, x_test_poly)

    def SVR(self, kernel):
        '''
        :param X: Input data
        :param y: Actual output
        :param X_test: Test data
        :param kernel: kernel used initially linear
        :return: model used and Predicted data
        '''
        regression = svm.SVR()
        train_time = 0

        if kernel == 'rbf':
            regression = svm.SVR(kernel='rbf')
            return self.run_model(regression, str(regression), self.X_train, self.y_train, self.x_test)

        elif kernel == 'poly':
            regression = svm.SVR(kernel='poly')
            return self.run_model(regression, str(regression), self.X_train, self.y_train, self.x_test)
        else:
            regression = svm.SVR(kernel='linear')
            return self.run_model(regression, str(regression), self.X_train, self.y_train, self.x_test)


    def Bayesian_Ridge(self):
        '''
        :param X: Input data
        :param y: Actual output
        :param X_test: Test data
        :return: model used and Predicted data
        '''
        regression = linear_model.BayesianRidge()
        return self.run_model(regression, str(regression), self.X_train, self.y_train, self.x_test)

    def Lasso_Regression(self):
        '''
        :param X: Input data
        :param y: Actual output
        :param X_test: Test data
        :return: model used and Predicted data
        '''
        regression = linear_model.Lasso()
        return self.run_model(regression, str(regression), self.X_train, self.y_train, self.x_test)

    def Ridge_Regression(self):
        '''
        :param X: Input data
        :param y: Actual output
        :param X_test: Test data
        :return: model used and Predicted data
        '''
        regression = linear_model.Ridge()
        return self.run_model(regression, str(regression), self.X_train, self.y_train, self.x_test)

    def Decision_tree_regressor(self, max_depth=3):
        '''
        :param X: Input data
        :param y: Actual output
        :param X_test: Test data
        :return: model used and Predicted data
        '''
        regression = DecisionTreeRegressor(max_depth=max_depth)
        return self.run_model(regression, str(regression), self.X_train, self.y_train, self.x_test)


    def Random_forest_regressor(self):
            '''
            :param X: Input data
            :param y: Actual output
            :param X_test: Test data
            :return: model used and Predicted data
            '''
            regression = RandomForestRegressor()
            return self.run_model(regression, str(regression), self.X_train, self.y_train, self.x_test)


    def run_model(self, model, model_name, x_train, y_train, x_test):
            train_time = 0
            if (os.path.exists("Saved_Models/Regression/" + model_name.split('(')[0] + ".pkl")):
                model = pickle.load(open("Saved_Models/Regression/" + model_name.split('(')[0] + ".pkl", 'rb'))
            else:
                start_time = time.time()
                model = model.fit(x_train, y_train)
                train_time = round((time.time() - start_time), 4)
                pickle.dump(model, open("Saved_Models/Regression/" + model_name.split('(')[0] + ".pkl", 'wb'))
            start_time = time.time()
            y_predict =[]
            if not len(x_test)==0:
                y_predict = model.predict(x_test)
            test_time = round((time.time() - start_time), 4)
            return y_predict, train_time, test_time

