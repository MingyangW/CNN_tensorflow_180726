# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 20:20:10 2018

@author: mingyang.wang
"""


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
 
# 其他库
import numpy as np
import matplotlib.pyplot as plt

 
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
 
#分类标签
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#单张图像展示，推荐使用python3
plt.figure()
plt.imshow(train_images[0])
#添加颜色渐变条
plt.colorbar()
#不显示网格线
plt.gca().grid(False)
 
#图像预处理
train_images = train_images / 255.0
test_images = test_images / 255.0
 
#样本展示
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
 
#检测模型    
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])    
 
model.compile(optimizer=tf.train.AdamOptimizer(), 
          loss='sparse_categorical_crossentropy', #多分类的对数损失函数
          metrics=['accuracy']) #准确度
 
model.fit(train_images, train_labels, epochs=5)
 
predictions = model.predict(test_images)
 
#前25张图分类效果
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
      color = 'green'
    else:
      color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label], 
                                  class_names[true_label]),
                                  color=color)
    
#单个图像检测
img = test_images[0]
print(img.shape) #28X28
 
#格式转换
img = (np.expand_dims(img,0))
print(img.shape) #1X28X28
 
predictions = model.predict(img)
prediction = predictions[0]
np.argmax(prediction) #9