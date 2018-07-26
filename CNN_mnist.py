#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 15:12:55 2018

@author: apple
"""

import tensorflow as tf

#计算准确率
def Accuracy(x, y):
    global prediction
    pre = sess.run(prediction, feed_dict={x_data:x})
    correct_pre = tf.equal(tf.arg_max(pre, 1), tf.arg_max(y, 1))
    accur = tf.reduce_mean(tf.cast(correct_pre, dtype=tf.float32))
    result = sess.run(accur, feed_dict={x_data:x, y_data:y})
    return result

#下载MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

#创建输入
x_data = tf.placeholder(tf.float32, [None, 28*28])
y_data = tf.placeholder(tf.float32, [None, 10])

x_image = tf.reshape(x_data, [-1, 28, 28, 1])
#创建卷积层1
W_conv_1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv_1 = tf.Variable(tf.constant(0.1, shape=[32]))
h_conv_1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv_1)
pool_1 = tf.nn.max_pool(h_conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#创建卷积层1
W_conv_2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv_2 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, W_conv_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv_2)
pool_2 = tf.nn.max_pool(h_conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

pool_2_reshape = tf.reshape(pool_2, [-1, 7*7*64])
#创建隐藏层
W_3 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
b_3 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_3 = tf.nn.relu(tf.matmul(pool_2_reshape, W_3) + b_3)

#创建全连接层
W_4 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_4 = tf.Variable(tf.constant(0.1, shape=[10]))
prediction = tf.nn.softmax(tf.matmul(h_3, W_4) + b_4)

#损失函数
loss = tf.reduce_mean(-tf.reduce_sum(y_data * tf.log(prediction), reduction_indices=[1]))
#训练
train = tf.train.AdamOptimizer(1e-4).minimize(loss)

#变量初始化
init = tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init)
    for i in range(501):
        x_batch, y_batch = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x_data:x_batch, y_data:y_batch})
        if i%50==0:
            print(Accuracy(mnist.test.images, mnist.test.labels))


