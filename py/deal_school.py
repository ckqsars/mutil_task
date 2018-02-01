#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2017 R&D, CINS Inc. (cins.com)
#
# Author: chenkaiqi<ckqcins@gmail.com>
#


import latent_multi_task as lt
import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch
import os
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge
import sklearn.metrics as SM

inf = float('inf')

cur_dir = os.path.dirname(os.path.abspath(__file__))
path_prefix = os.path.abspath(os.path.join(cur_dir, "../"))




def load_school_data():
    """Load School data set.

    The data set's properties are:
    - 15362 samples
    - 28 features (original features were:
        - year of examination,
        - 4 school-specific features,
        - 3 student-specific features.
        Categorical features were replaced with a binary feature for each
        possible feature value. In total this resulted in 27 features. The last
        feature is the bias term)
    - regression task of predicting student's exam score

    Note
    ----
    The data set is the one provided by "Argyriou, Evgeniou, Pontil - Convex
    multi-task feature learning - ML 2008" on their web site.

    Returns
    -------
    tasks -- list
        A list of Bunch objects that correspond to regression tasks, each task
        corresponding to one school.

    """
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
    # indexing
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
    m = 0
    for i in range(len(tasks)):
        tempX = tasks[i]['data']
        tempY = tasks[i]['target']
        # print tempY.shape
        tempX = np.column_stack((tasks[i]['data'],   np.ones((tempX.shape[0],1))*(i+1)))
        tempY = np.column_stack((tasks[i]['target'], np.ones((tempY.shape[0],1))*(i+1)))
        # print tempY.shape
        # if tempY.shape[0] < 50:
        #     m = m +1
        # if i == 0 :
        #     X = tempX
        #     Y = tempY
        # else:
        #     X = np.row_stack((X,tempX))
        #     Y = np.row_stack((Y,tempY))

    # print m

    return tasks,X


def load_school_dataset():
    """
    Load School dataset and select the first 27 tasks  for computing reasons
    """
    dataset = sio.loadmat('../data/school1.mat')
    FEATURES_COLUMNS = ['Year_1985', 'Year_1986', 'Year_1987', 'FSM', 'VR1Percentage', 'Gender_Male', 'Gender_Female',
                        'VR_1', 'VR_2', 'VR_3',
                        'Ethnic_ESWI', 'Ethnic_African', 'Ethnic_Arabe', 'Ethnic_Bangladeshi', 'Ethnic_Carribean',
                        'Ethnic_Greek', 'Ethnic_Indian',
                        'Ethnic_Pakistani', 'Ethnic_Asian', 'Ethnic_Turkish', 'Ethnic_Others', 'SchoolGender_Mixed',
                        'SchoolGender_Male',
                        'SchoolGender_Female', 'SchoolDenomination_Maintained', 'SchoolDenomination_Church',
                        'SchoolDenomination_Catholic',
                        'Bias']

    # Dataframe representation
    X_df = pd.DataFrame(dataset['X'][:, 0][0], columns=FEATURES_COLUMNS)
    y_df = pd.DataFrame(dataset['Y'][:, 0][0], columns=['Exam_Score'])
    X_df['School'] = 1
    y_df['School'] = 1

    d = X_df.shape[1] - 1
    for i in range(1, d):
        X_df_i = pd.DataFrame(dataset['X'][:, i][0], columns=FEATURES_COLUMNS)
        X_df_i['School'] = i + 1
        X_df = X_df.append(X_df_i, ignore_index=True)

        y_df_i = pd.DataFrame(dataset['Y'][:, i][0], columns=['Exam_Score'])
        y_df_i['School'] = i + 1
        y_df = y_df.append(y_df_i, ignore_index=True)

    return X_df.values, y_df.values


def main():

    data_dict = {}
    X,Y = load_school_dataset()

    result,XT = load_school_data()


    t= 0
    f = 0
    sum_nmse = 0
    score = 0
    x = []
    y = []
    for i in range(len(result)):

        model = lt.multi_task(28)
        x = result[i]['data']
        y = result[i]['target']
        # x.extend(result[i]['data'])
        # y.extend(result[i]['target'])
        # train_X, test_X, train_data, test_data = model.split_data(x, y, 0.2)
        train_X, test_X, train_data, test_data = train_test_split(x, y, test_size=0.9)
        w = model.training_data(train_X, train_data)
        # print w.shape
        test_y = model.prediction(test_X)
        data_dict[i] = {'train_data': train_data, 'test_data':test_data,'train_X': train_X,\
                        'w_': w, 'test_X':test_X}
        sum_nmse = model.nmseEval(test_data, test_y) + sum_nmse
        score = model.score(test_X, test_data) + score
        # print len(x)
    # test_data = list(np.array(test_data))
    # test_y = list(test_y)
    # print test_y.shape

    # print sum_nmse/139
    si_sum_nmse = sum_nmse/139
    # print score/139
    #     # print sum_nmse
    #     # t = t + len(school['data'])
    #     # f = f + len(school['target'])
    #     # print school['target']
    # print type(sum_nmse)
    # print f

    # print x.shape
    # print y.shape

    for t in range(1):
        lambda_dict = {}
        explain_var = 0
        for index in data_dict:
            # print index
            lambda_dict[index] = {}
            w = data_dict[index]['w_']
            for index1 in data_dict:
                t = data_dict[index1]['train_data']
                x = data_dict[index1]['train_X']
                lambda_value = get_lambda(w, x, t)
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
                    new_x = get_new_x(lambda_value, temp_train_x)
                    train_x = np.matrix(np.vstack((np.array(train_x), np.array(new_x))))
                    temp_train_data = data_dict[index1]['train_data']
                    train_y = np.matrix(np.vstack((np.array(train_y), np.array(temp_train_data))))

            # print train_y.shape
            # print train_x.shape
            # clf = linear_model.LinearRegression()
            clf = Ridge(alpha=1)
            clf.fit(train_x, train_y)
            w_ = clf.coef_
            data_dict[index]['w_'] = w_
            score = model.score(test_X, test_y)
            y = np.matrix(clf.predict(test_X))
            sum_score = sum_score + score
            # if index == 0:
            #     print clf.coef_
            sum_nmse = model.nmseEval(y, test_y) + sum_nmse
            explain_var = SM.explained_variance_score(test_y, y) + explain_var

        sum_score = sum_score /139
        # print sum_score
        # print sum_nmse / 139
        ml_sum_nmse = sum_nmse/139
        # print explain_var/100
        # print data_dict[0]['w_']
    return si_sum_nmse, ml_sum_nmse

def get_lambda(w, x, t):
    model = lt.multi_task(28)
    temp = np.diag(w.flat)
    # print temp
    s1 = np.matrix(x) * temp

    lambd_value = model.training_data(s1, t,
                                      model_name='Ridge')
    # print lambd_value.shape
    # # print s1.shape
    # s2 = s1.T*s1
    # s3 = np.linalg.inv(s2)
    # lambd_value = s3*s1.T*t
    # print lambd_value.shape
    return lambd_value.T


def get_new_x(lambda_value, x):
    temp = np.diag(lambda_value.flat)
    new_x = np.matrix(x) * temp

    return new_x



if __name__ == '__main__':

    st = 0
    mt = 0
    t = 10
    while(t):
        st_score,ml_score = main()
        if st_score == inf:
            continue

        else:
            t =t - 1
            print t
            print "st_score:{0},ml_score{1}".format(st_score,ml_score)
            st = st + st_score
            mt = mt + ml_score

        break

    print st/10
    print mt/10

    # main()