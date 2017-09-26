#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2017 R&D, CINS Inc. (cins.com)
#
# Author: chenkaiqi<ckqcins@gmail.com>
#



import latent_multi_task






def main():
    model = latent_multi_task.multi_task(80)
    train_data, test_data, train_X, w_, test_X, score = model.simulation(3)
    print score
    print w_.size



if __name__ == "__main__":
    main()