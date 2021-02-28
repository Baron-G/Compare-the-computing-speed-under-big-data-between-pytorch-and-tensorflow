#!/usr/bin/env python
# coding: utf-8

# In[11]:


import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import time
import matplotlib.pyplot as plt
from sklearn import datasets


# In[2]:


#准备数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(np.argmax(y_test[100]))
#print(y_test[0])
#x_train=x_train.reshape(60000,784)
#x_test=x_test.reshape(10000,784) 


# In[3]:


# 输入占位符
tf.compat.v1.disable_eager_execution()
xtr = tf.placeholder("float32", [None,25,25])
xte = tf.placeholder("float32", [25,25])
#xtr = tf.placeholder("float32", [None,784])
#xte = tf.placeholder("float32", [784])
#x_train = tf.cast(x_train, tf.float32)
#x_test=tf.cast(x_train, tf.float32)
#print(x_test.shape[0])
#plt.imshow(x_test[1,:,:], cmap="binary")
#plt.show()


# In[4]:


#sess=tf.Session()
#init=tf.global_variables_initializer()
#sess.run(init)
#distance = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.pow(tf.add(x_train, tf.negative(x_test[1])),2),reduction_indices=1),reduction_indices=1))
#pred = tf.arg_min(distance, 0)
#print(sess.run(distance))


# In[5]:


# 计算L2距离
distance = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.pow(tf.add(xtr, tf.negative(xte)),2),reduction_indices=1),reduction_indices=1))
#distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.add(xtr, tf.negative(xte)),2),reduction_indices=1))
# 获取最小距离的索引
pred = tf.arg_min(distance, 0)


# In[6]:


# 初始化变量
init = tf.global_variables_initializer()


# In[9]:


number=1000
#分类精确度
accuracy = 0
# 运行会话，训练模型
start = time.perf_counter()
with tf.Session() as sess:
    # 运行初始化
    sess.run(init)
    # 遍历测试数据
    for i in range(number):
        # 获取当前样本的最近邻索引
        nn_index = sess.run(pred, feed_dict={xtr: x_train, xte: x_test[i]})   #向占位符传入训练数据
        #print(nn_index)
        # 最近邻分类标签与真实标签比较
        #if i %50 == 0:
            #print("Test", i, "Prediction:", y_train[nn_index], \"True Class:", y_test[i])
        # 计算精确度
        if y_train[nn_index] == y_test[i]:
            accuracy += 1/number

    print("Done!")
    print(f"Accuracy:{accuracy}")
elapsed = (time.perf_counter()-start)
print(f'time use :{elapsed}')


# In[8]:


#print(x_test)
#plt.imshow(x_test[1], cmap="binary")
#plt.show()


# In[ ]:




