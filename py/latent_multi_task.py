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

class multi_task(object):

    def __init__(self,feature_dim = 80):
        self.feature_dim = feature_dim

    def simulation(self, n):
        train_data, test_data, train_X, test_X, W = self.create_dataS1(n)
        clf = linear_model.LinearRegression()
        clf.fit(train_X, train_data)
        w_ = clf.coef_
        #print w_
        score = clf.score(test_X, test_data)
        train_y = clf.predict(train_X)

        return train_data, test_data, train_X, W, test_X, score

    def create_dataS1(self, n):
        X = np.random.normal(0, 1, (400, self.feature_dim-1))
        x0 = np.ones((400,1))
        X = np.hstack((X,x0))
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



