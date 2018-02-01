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
    matlab_file = os.path.join(path_prefix, "data/school_splits/school_b.mat")
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