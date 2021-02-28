#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow.compat.v1 as tf
# 使用Eager Execution动态图机制
tf.enable_eager_execution()
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import time


# In[2]:


iris = datasets.load_iris(return_X_y=False)


# In[3]:


def pca_svd(x,dim = 2):
    '''
        x:输入矩阵
        dim:降维之后的维度数
    '''
    with tf.name_scope("SVD"):#创建一个参数名空间
        m,n= tf.to_float(x.get_shape()[0]),tf.to_int32(x.get_shape()[1])
        assert not tf.assert_less(dim,n)
        mean= tf.reduce_mean(x,0)
        # 去中心化 防止数据过分逼近大的值 而忽略小的值
        x_new = x - mean
        #print(x_new)
        # 计算奇异值
        s, u, v = tf.linalg.svd(x_new)
        pca = tf.matmul(x_new,v[:,:dim])
        #print(v[:,:dim])
        #print(v)
    return pca


# In[4]:


start = time.perf_counter()
pca_data = tf.constant(np.reshape(iris.data,(iris.data.shape[0],-1)),dtype=tf.float32)
pca_data = pca_svd(pca_data,dim=2)
elapsed = (time.perf_counter()-start)
print(f'time use :{elapsed}')
#print(pca_data)


# In[5]:


Y= iris.target
pca = pca_data.numpy()
plt.figure()
color=['red','green','blue']
for i, target_name in enumerate(iris.target_names):
    plt.scatter(pca[Y==i,0],pca[Y==i,1],label = target_name, color = color[i])
plt.legend()
plt.title('svd')
plt.show


# In[6]:


n0=2*np.random.randn(5000,1000)+1
n1=-2*np.random.randn(5000,1000)-1
n=np.vstack((n0,n1))

start = time.perf_counter()
pca_data = tf.constant(np.reshape(n,(n.shape[0],-1)),dtype=tf.float32)
pca_data = pca_svd(pca_data,dim=2)
elapsed = (time.perf_counter()-start)
print(f'time use :{elapsed}')


# In[ ]:




