#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2017 R&D, CINS Inc. (cins.com)
#
# Author: chenkaiqi<ckqcins@gmail.com>
#



import latent_multi_task as lt
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import Ridge
import sklearn.metrics as SM


model = lt.multi_task(80)



def main():

    data_dict = {}
    sum_score = 0
    sum_nmse = 0
    explain_var = 0
    for i in range(100):

        train_data, test_data, train_X, w_, test_X, score,test_y = model.simulation(3)
        data_dict[i] = {'train_data': train_data, 'test_data':test_data,'train_X': train_X,\
                        'w_': w_, 'test_X':test_X, 'score' : score}

        sum_score = sum_score + score
        # print test_y.shape
        # print test_data.shape
        sum_nmse = model.nmseEval(test_y,test_data) + sum_nmse
        explain_var =   SM.explained_variance_score(test_data,test_y) + explain_var

    score = sum_score/100
    sum_nmse = sum_nmse/100
    explain_var = explain_var/100

    # print explain_var
    # print score
    print sum_nmse
    print data_dict[0]['train_data'].shape
    print data_dict[0]['train_X'].shape
    print data_dict[0]['w_'].shape
    print type(data_dict[0]['train_data'])
    print data_dict[0]['w_']
    print test_y.shape
    print type(test_y)

    
    for t  in range(3):
        lambda_dict = {}
        explain_var = 0
        for index in data_dict:
            lambda_dict[index] = {}
            w = data_dict[index]['w_']
            for index1 in data_dict:
                t = data_dict[index1]['train_data']
                x = data_dict[index1]['train_X']
                lambda_value=get_lambda(w,x,t)
                # print lambda_value
                # print lambda_value.shape
                lambda_dict[index][index1] = lambda_value
        sum_score = 0
        sum_nmse = 0
        for index in data_dict:
            train_x = data_dict[index]['train_X']
            train_y = data_dict[index]['train_data']
            test_X = data_dict[index]['test_X']
            test_y = data_dict[index]['test_data']
            for index1 in data_dict:
                if index1 != index:
                    lambda_value = lambda_dict[index][index1]
                    temp_train_x = data_dict[index1]['train_X']
                    new_x = get_new_x(lambda_value,temp_train_x)
                    train_x = np.matrix(np.vstack((np.array(train_x),np.array(new_x))))
                    temp_train_data = data_dict[index1]['train_data']
                    train_y = np.matrix(np.vstack((np.array(train_y),np.array(temp_train_data))))

            # print train_y.shape
            # print train_x.shape
            # clf = linear_model.LinearRegression()
            clf = Ridge(alpha=1)
            clf.fit(train_x, train_y)
            w_ = clf.coef_
            data_dict[index]['w_'] = w_
            score = clf.score(test_X, test_y)
            y = np.matrix(clf.predict(test_X))
            sum_score = sum_score + score
            # if index == 0:
            #     print clf.coef_
            sum_nmse = model.nmseEval(y, test_y) + sum_nmse
            explain_var = SM.explained_variance_score(test_y,y) + explain_var

        sum_score = sum_score/100
        # print sum_score
        print sum_nmse/100
        # print explain_var/100
        # print data_dict[0]['w_']



def get_lambda(w,x,t):
    temp = np.diag(w.flat)
    # print temp
    s1 = x*temp

    lambd_value = model.training_data(s1,t,
                                      model_name='Ridge')
    # print lambd_value.shape
    # # print s1.shape
    # s2 = s1.T*s1
    # s3 = np.linalg.inv(s2)
    # lambd_value = s3*s1.T*t
    # print lambd_value.shape
    return lambd_value.T

def get_new_x(lambda_value,x):
    temp = np.diag(lambda_value.flat)
    new_x = x*temp

    return new_x

if __name__ == "__main__":
    main()