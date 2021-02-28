#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
from sklearn import datasets
import time 


# In[2]:


x0=np.random.randn(50,1)
x1=np.random.randn(50,1)
x2=2*x0+2*np.random.rand(50,1)+1
x3=2*x1-2*np.random.rand(50,1)-1
y0=np.ones((50,1))
y1=-y0
n=np.vstack((x0,x1))
m=np.vstack((x2,x3))
x_train=np.hstack((n,m))
x_target=np.vstack((y0,y1))
#plt.plot(x_train[:,0],x_train[:,1],"ob")
#x = np.arange(-3,4) 
#p = 2*x
#plt.plot(x,p) 
#plt.show()


# In[3]:


tf.compat.v1.disable_eager_execution()
#X=tf.constant(x_train,dtype=tf.float32)
#Y=tf.constant(x_target,dtype=tf.float32)
#初始化feedin
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)


# In[4]:


#创建变量
A=tf.Variable(tf.random_normal(shape=[2,1]))
b=tf.Variable(tf.random_normal(shape=[1,1]))


# In[5]:


# 定义线性模型
model_output = tf.subtract(tf.matmul(x_data, A),b)


# In[6]:


#计算w^Tw
L2 = tf.matmul(tf.transpose(A),A)
#print(L2)


# In[7]:


#LOSS= L2+C*max(0,1-pred)
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))
Loss = tf.add(classification_term,L2)


# In[8]:


my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(Loss)
init = tf.global_variables_initializer()


# In[9]:


start = time.perf_counter()
with tf.Session() as sess:
    sess.run(init)
    for i in range(20000):
        sess.run(train_step,feed_dict = {x_data: x_train , y_target: x_target})
    [[a1], [a2]] = sess.run(A)
    [[b]] = sess.run(b)
elapsed = (time.perf_counter()-start)
#print(a1)
print(f'time use :{elapsed}')


# In[10]:


plt.plot(x_train[:50,0],x_train[:50,1],"ob")
plt.plot(x_train[50:,0],x_train[50:,1],"or")
p = -(a2/a1)*x_train[:,0]-b/a1
p2=p-1
p3=p+1
x = np.arange(-3,4) 
plt.plot(x_train[:,0],p)
plt.plot(x_train[:,0],p2)
plt.plot(x_train[:,0],p3)
plt.show()

