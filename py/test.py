#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2017 R&D, CINS Inc. (cins.com)
#
# Author: chenkaiqi<ckqcins@gmail.com>
#


import numpy as np
import pandas as pd
import scipy.io as sio

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
    X,y = load_school_dataset()


if __name__== "__main__":

   main()