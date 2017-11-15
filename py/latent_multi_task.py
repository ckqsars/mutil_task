#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2017 R&D, CINS Inc. (cins.com)
#
# Author: chenkaiqi<ckqcins@gmail.com>
#

import random
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import Ridge

class multi_task(object):

    def __init__(self,feature_dim = 80):
        self.feature_dim = feature_dim

    def simulation(self, n):
        train_data, test_data, train_X, test_X, W = self.create_dataS1(n)
        # clf = linear_model.LinearRegression()
        clf = Ridge(alpha=0.6)
        clf.fit(train_X, train_data)
        w_ = clf.coef_

        # w = self.calu_W(train_data, train_X)
        # clf.coef_ = w.T

        score = clf.score(test_X, test_data)
        test_y = np.matrix(clf.predict(test_X))
        # score = clf.score(train_X,train_data)
        # print clf.coef_
        return train_data, test_data, train_X, clf.coef_, test_X, score,test_y

    def calu_W(self,data, X):
        s1 = X.T * X
        s2 = np.linalg.inv(s1)
        s3 = s2 * X.T
        W = s3 * data

        return W

    def create_dataS1(self, n):
        X = np.random.normal(0, 1, (400, self.feature_dim))
        # x0 = np.ones((400,1))
        # X = np.hstack((X,x0))
        w = np.random.uniform(0, 10, (1, self.feature_dim))
        X = np.mat(X)
        w = np.mat(w)
        Y = X * w.T
        noise = np.random.normal(0, 10, (400, 1))
        data = Y + np.mat(noise)
        test_data = data[n * data.shape[0] / 10:]
        train_data = data[:n * data.shape[0] / 10]
        test_X = X[n * X.shape[0] / 10:]
        train_X = X[:n * X.shape[0] / 10]

        return train_data, test_data, train_X, test_X, w

    def NMSE(self, y, y_):
        return (y - y_).T * (y - y_) / (y_.T * y_)



