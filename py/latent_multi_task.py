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
from sklearn.metrics import mean_squared_error
import math
import random
import math

class multi_task(object):

    def __init__(self,feature_dim = 80):
        self.feature_dim = feature_dim

    def simulation(self, n):
        train_data, test_data, train_X, test_X, W = self.create_dataS1(n)
        # clf = linear_model.LinearRegression()
        clf = Ridge(alpha=1)
        clf.fit(train_X, train_data)
        w_ = clf.coef_



        score = clf.score(test_X, test_data)
        test_y = np.matrix(clf.predict(test_X))
        # score = clf.score(train_X,train_data)
        # print clf.coef_
        return train_data, test_data, train_X, clf.coef_, test_X, score,test_y

    def create_dataS1(self, n):
        X = np.random.normal(0, 1, (400, self.feature_dim-1))
        # x0 = np.ones((400,1))
        # X = np.hstack((X,x0))
        w = np.random.uniform(0, 10, (1, self.feature_dim-1))
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

    def computeNMSE(self, estBeta, origBeta):
        estBetaSum = sum(estBeta)
        origBetaSum = sum(origBeta)

        nmse = 0
        for i in xrange(len(estBeta)):
            nmse += ((estBeta[i] / estBetaSum) - (origBeta[i] / origBetaSum)) ** 2

        return nmse / len(estBeta)

    def score(self, X, y):
        y_pred = self.prediction(X)
        return 1.- np.sqrt(mean_squared_error(y[:,0], y_pred[:,0]))/(np.max(y[:,0])-np.min(y[:,0]))

    def nmseEval(self, predictionData,targetData):
        # print(targetData)
        # print(predictionData)
        nmse = 0.0
        avgTarget = sum(targetData)/len(targetData)
        avgPrediction = sum(predictionData)/len(predictionData)
        var = 0
        new_var = targetData.var()
        for i in range(0, len(targetData)):
            nmse = nmse + math.pow(targetData[i] - predictionData[i], 2.0)
            var = math.pow(targetData[i] - avgTarget, 2.0) + var
        #     avgTarget = avgTarget + targetData[i]
        #     avgPrediction = avgPrediction + predictionData[i]
        #
        # avgTarget = avgTarget / float(len(targetData))
        # avgPrediction = avgPrediction / float(len(targetData))
        #     nmse1 = nmse
        # nmse = nmse / (len(targetData) * avgTarget * avgPrediction)
        # print nmse
        # print var

        nmse = math.sqrt(nmse) / (new_var*len(targetData))
        # print var
        # print new_var
        # print type(nmse)

    #     if nmse > 1.0:
    #         print("NMSE larger then 1.0")
    #         print("avgTarget: " + str(avgTarget))
    #         print("avgPrediction: " + str(avgPrediction))
    #         print("original nmse: " + str(nmse1))
    #         print("len(targetData): " + str(len(targetData)))
    #     return ["nmse", nmse]
        return nmse

    def training_data(self,x,y, model_name = 'Ridge'):
        assert model_name in ['Ridge','lr']
        if model_name == 'Ridge':
            self.model = Ridge(alpha = 1)

        self.model.fit(x,y)
        w = self.model.coef_

        return w

    def prediction(self,test_X):
        test_y = np.matrix(self.model.predict(test_X))

        return test_y






    def split_data(self,feature,target,split_ratio):



        '''

        :param feature: the array or list of the feature in dataset
        :param target:  the array or list of the target in dataset
        :split_ratio:   the ratio of test_data in dataset
        :return: train_data, test_data, train_target, test_target


        '''
        assert type(feature) in [list, np.ndarray]

        if type(feature) == list:
            if len(feature) != len(target):
                print "error in data"

                return 0

            len_dataset = len(feature)
            len_test = int(len_dataset*split_ratio)
            test_data = []
            test_target = []
            while(len_test):
                random_num = int(len_dataset*random.random())
                test_data.append(feature[random_num])
                test_target.append(target[random_num])
                feature.remove(feature[random_num])
                target.remove(target[random_num])
                len_dataset = len_dataset - 1
                len_test = len_test - 1
            train_data =feature.copy()
            train_target = target.copy()

        else:
            # test_data = np.array([])
            # test_target = np.array([])
            len_dataset = feature.shape[0]
            len_test = int(len_dataset*split_ratio)
            while(len_test):
                random_num = int(len_dataset * random.random())
                if len_test == int(len_dataset*split_ratio):
                    test_data = feature[random_num]
                    test_target = target[random_num]
                else:
                    test_data = np.row_stack((test_data, feature[random_num]))
                    test_target = np.row_stack((test_target,target[random_num]))
                feature = np.delete(feature,random_num,axis=0)
                target = np.delete(target,random_num,axis=0)
                len_dataset = len_dataset -1
                len_test = len_test - 1
            train_data = feature
            train_target = target



        return train_data, test_data, train_target, test_target
