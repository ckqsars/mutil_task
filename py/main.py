#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2017 R&D, CINS Inc. (cins.com)
#
# Author: chenkaiqi<ckqcins@gmail.com>
#



import latent_multi_task






def main():

    data_dict = {}
    sum_score = 0
    for i in range(100):
        model = latent_multi_task.multi_task(80)
        train_data, test_data, train_X, w_, test_X, score = model.simulation(3)
        data_dict[i] = {'train_data': train_data, 'test_data':test_data,'train_X': train_X,\
                        'w_': w_, 'test_X':test_X, 'score' : score}
        
        sum_score = sum_score + score
    score = sum_score/100
    print score



if __name__ == "__main__":
    main()