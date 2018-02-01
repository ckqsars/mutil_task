#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2017 R&D, CINS Inc. (cins.com)
#
# Author: chenkaiqi<ckqcins@gmail.com>
#

import tensorflow as tf
import math



batch_size = 20
checkpoint_dir = './checkpoint'
n_input = 28

class model():
    def __init__(self):

        self.X = tf.placeholder('float',[None,n_input])

        self.y = tf.placeholder('float',[None,1])


    def build_model(self,n_input,):



        hidden1_units = int(math.sqrt(n_input))
        hidden2_units = int(math.sqrt(n_input))
        with tf.name_scope('hidden1'):
            weights = tf.Variable(
                tf.truncated_normal([n_input, hidden1_units],
                                    stddev=1.0 / math.sqrt(float(n_input))),
                name='weights')  # 权重是标准方差为输入尺寸开根号分之一的正态分布
            biases = tf.Variable(tf.zeros([hidden1_units]),
                                 name='biases')
            hidden1 = tf.nn.relu(tf.matmul(self.X, weights) + biases)
            # Hidden 2
        with tf.name_scope('hidden2'):
            weights = tf.Variable(
                tf.truncated_normal([hidden1_units, hidden2_units],
                                    stddev=1.0 / math.sqrt(float(hidden1_units))),
                name='weights')
            biases = tf.Variable(tf.zeros([hidden2_units]),
                                 name='biases')
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
            # Linear
        with tf.name_scope('full_connect'):
            weights = tf.Variable(
                tf.truncated_normal([hidden2_units, 1],
                                    stddev=1.0 / math.sqrt(float(hidden1_units))),
                name='weights')
            biases = tf.Variable(tf.zeros([1]),
                                 name='biases')
            self.pred = tf.matmul(hidden2, weights) + biases



    def build_loss(self):

        self.loss = tf.reduce_mean(tf.square(self.pred-self.y))




    def build_training(self, learning_rate, ):

        tf.summary.scalar('loss', self.loss)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        self.train_step = optimizer.minimize(self.loss, global_step=global_step)



    def out_model(self):

        out =  self.pred


    def train_model(self,X,Y):
        datasize = X.shape[0]
        saver = tf.train.Saver()
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True

        # with tf.Session(config=config) as sess:
        with tf.Session(config=tf.ConfigProto(device_count={'cpu':0})) as sess:
            sess.run(tf.global_variables_initializer())
            # ckpt = tf.train.latest_checkpoint()
            STEPS = 5000

            for i in range(STEPS):
                start = (i * batch_size) % datasize
                end = min(start + batch_size, datasize)
                # tf.convert_to_tensor()
                feed = {self.X: X[start:end], self.y: Y[start:end]}
                sess.run(self.train_step, feed_dict=feed)
                if i % 100 == 0:
                    print sess.run(self.loss, feed_dict=feed)
                    # correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
                    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            saver.save(sess, "checkpoint/model_dis2.ckpt")

    def infer(self,X):

        saver = tf.train.Saver()

        out = self.out_model()

        with tf.Session() as sess:
            saver.restore(sess, 'model/model_dis2.ckpt')
            result = sess.run(out,feed_dict={self.x:X})



        return result



