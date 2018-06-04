#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2017 R&D, CINS Inc. (cins.com)
#
# Author: chenkaiqi<ckqcins@gmail.com>
#

import numpy as np
import scipy.io as sio
import os
from sklearn.datasets.base import Bunch
import random


def _preprocessing(X,Y):
    """
    Prepare the dataset for the MTL algorithms
    """
    X_process=np.concatenate((X,np.ones((X.shape[0],1))),axis=1)
    y_process=np.concatenate((Y[:,0].reshape(Y.shape[0],1),np.ones((Y.shape[0],1))),axis=1)
    for l in range(2,Y.shape[1]+1):
        X_l=np.concatenate((X,np.ones((X.shape[0],1))*l),axis=1)
        X_process=np.append(X_process,X_l,axis=0)
        y_l = np.concatenate((Y[:,0].reshape(Y.shape[0],1),l*np.ones((Y.shape[0],1))),axis=1)
        y_process=np.append(y_process,y_l,axis=0)
    return X_process, y_process


def random_data():


    dim = 80
    data_dict = {}
    N = 100
    for i in range(N):
        X = np.random.normal(0, 1, (400, dim - 1))
        # x0 = np.ones((400,1))
        # X = np.hstack((X,x0))
        w = np.random.uniform(0, 10, (1, dim - 1))
        X = np.mat(X)
        w = np.mat(w)
        Y = X * w.T
        noise = np.random.normal(0, 10, (400, 1))
        data = Y + np.mat(noise)
        data_dict[i] = {'X':X,'y':data}

    return data_dict,dim,N


def load_school_data():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path_prefix = os.path.abspath(os.path.join(cur_dir, "../"))
    matlab_file = os.path.join(path_prefix, "data/school/school_b.mat")
    mat = sio.loadmat(matlab_file)
    # extract combined X and y data
    X = mat["x"].T.astype("float")
    y = mat["y"].astype("float")
    #    # DEBUG
    #    print "Inspection of values of features for School data"
    #    for i in range(X.shape[1]):
    #        col = X[:, i]
    #        if not np.all(np.unique(col) == np.array([0, 1])):
    #            print "Column {} has non-binary unique values: {}".format(i,
    #                                                                np.unique(col))
    # extract starting indices of tasks and subtract 1 since MATLAB uses 1-based
    # indexing    file_path = ''
    start_ind = np.ravel(mat["task_indexes"] - 1)
    # split the data to separate tasks
    tasks = []
    for i in range(len(start_ind)):
        start = start_ind[i]
        if i == len(start_ind) - 1:
            end = -1
        else:
            end = start_ind[i + 1]
        descr = "School data: school {}".format(i + 1)
        id = "School {}".format(i + 1)
        tasks.append(Bunch(data=X[start:end],
                           target=y[start:end],
                           DESCR=descr,
                           ID=id))
    data_dict = {}
    for i in range(len(tasks)):
        tempX = tasks[i]['data']
        tempY = tasks[i]['target']
        data_dict[i] = {'X':tempX, 'y':tempY}

        if i == 0 :
            X = tempX
            Y = tempY
        else:
            X = np.row_stack((X,tempX))
            Y = np.row_stack((Y,tempY))

    N = len(data_dict)
    dim = 28
    return data_dict, X, Y


def load_sarcos_dataset(set_size=1000):
    """
    Load SARCOS dataset and select the first 2000 samples for computing reasons
    """
    # Load training set
    sarcos_train = sio.loadmat('../data/sarcos_inv.mat')
    # Inputs  (7 joint positions, 7 joint velocities, 7 joint accelerations)
    Xtrain = sarcos_train["sarcos_inv"][:, :21]
    # Outputs (7 joint torques)
    Ytrain = sarcos_train["sarcos_inv"][:, 21:]

    # Load test set
    sarcos_test = sio.loadmat("../data/sarcos_inv_test.mat")
    Xtest = sarcos_test["sarcos_inv_test"][:, :21]
    Ytest = sarcos_test["sarcos_inv_test"][:, 21:]

    X = np.concatenate((Xtrain,Xtest),axis=0)
    Y = np.concatenate((Ytrain,Ytest),axis=0)

    return _preprocessing(X[:set_size,:],Y[:set_size,:])

