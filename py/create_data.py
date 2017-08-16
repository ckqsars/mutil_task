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
import matplotlib.pyplot as plt

def NMSE(y,y_):
    return (y-y_).T * (y-y_) / (y_.T * y_)

def create_dataS1():
    X = np.random.normal(0,1,(400,80))
    w = np.random.uniform(0,10,(1,80))
    X= np.mat(X)
    w = np.mat(w)
    Y = X*w.T
    noise = np.random.normal(0,10,(400,1))
    data = Y + np.mat(noise)
    n = 3
    # test_data  = data[:data.shape[0]/10]
    # train_data = data[data.shape[0]/10:]
    # test_X = X[:X.shape[0]/10]
    # train_X = X[X.shape[0]/10:]
    test_data  = data[n*data.shape[0]/10:]
    train_data = data[:n*data.shape[0]/10]
    test_X = X[n*X.shape[0]/10:]
    train_X = X[:n*X.shape[0]/10]


    return train_data, test_data, train_X, test_X, w

def calu_W(data,X):
    s1 = X.T*X
    s2 = np.linalg.inv(s1)
    s3 = s2*X.T
    W = s3*data

    return W

def Calu_Distance(y, y_):
    return np.square(y-y_)

def Calu_Error(y, y_):
    #print y.size
    return (np.sum(Calu_Distance(y, y_))) / y.size

def Get_Noise(X,distanceYY_,data):
    '''

    :param X: the mat of feature of data
    :param distanceYY_:  the distance between the y_ and y the y_ is the reality data
    :param data: the T of X
    :return: the noise_X to others model.
    '''
    distance_X = np.array(np.hstack((distanceYY_,np.hstack((X,data)))))
    #print distance_X
    distance_X = distance_X[np.lexsort(-distance_X[:,::-1].T)]
    noise_X = distance_X[:distanceYY_.size/10,1:]
    return np.mat(noise_X)




def main():
    data_Dict = {}
    noise_dataList = []
    sum_cost = 0
    sum_nmse = 0
    for i in range(100):
        clf = linear_model.LinearRegression()
        train_data, test_data, train_X, test_X, w_ = create_dataS1()
        clf.fit(train_X, train_data)
        W = calu_W(train_data, train_X)
        train_y = clf.predict(train_X)
        score = clf.score(test_X, test_data)
        test_y = clf.predict(test_X)
        #print test_y
        nMSE = NMSE(test_y, test_data)
        sum_nmse = nMSE + sum_nmse
        # print score
        # clf.coef_ = W.T
        # score = clf.score(test_X, test_data)
        # # print score
        distanceYY_ = Calu_Distance(train_data, train_y)
        noise_X = Get_Noise(train_X, distanceYY_, train_data)
        data_Dict[i] = {'train_data': train_data, 'test_data':test_data,'train_X': train_X,\
                        'w_': w_, 'test_X':test_X, 'score' : score}
        noise_dataList.append(noise_X)
        sum_cost = sum_cost + score
        # sum_cost1 = sum_cost1 + score1

    print sum_nmse/100
    print sum_cost/100
    new_sum = 0
    new_sum_nmse = 0
    for i in range(100):
        train_X = data_Dict[i]['train_X']
        train_data = data_Dict[i]['train_data']
        test_data = data_Dict[i]['test_data']
        test_X = data_Dict[i]['test_X']
        index = int(random.random()*100)
        noise_X = noise_dataList[index]
        train_X = np.vstack((train_X, noise_X[:,:-1]))
        train_data = np.vstack((train_data, noise_X[:,-1:]))
        clf = linear_model.LinearRegression()
        clf.fit(train_X, train_data)
        newScore = clf.score(test_X, test_data)
        data_Dict[i]['new_Score'] = newScore
        new_sum = new_sum + newScore
        test_y = clf.predict(test_X)
        nMSE = NMSE(test_y, test_data)
        new_sum_nmse = nMSE + new_sum_nmse

    print new_sum / 100
    print new_sum_nmse/100

    # for index in data_Dict:
    #     print data_Dict[index]['score'] - data_Dict[index]['new_Score']

    #print noise_dataList
    # plt.plot(data,'b')
    # plt.plot(y,'r')
    # plt.show()


if __name__ == "__main__":
    main()

