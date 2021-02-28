#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow.compat.v1 as tf
# 使用Eager Execution动态图机制
tf.enable_eager_execution()
import numpy as np
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
import time


# In[2]:


data = datasets.load_iris(return_X_y=False)
n0=2*np.random.randn(5000,1000)+1
n1=-2*np.random.randn(5000,1000)-1
n=np.vstack((n0,n1))


# In[3]:


def pca(x,dim = 2):
    '''
        x:输入矩阵
        dim:降维之后的维度数
    '''
    with tf.name_scope("PCA"):#创建一个参数名空间
        m,n= tf.to_float(x.get_shape()[0]),tf.to_int32(x.get_shape()[1])
        assert not tf.assert_less(dim,n)
        mean = tf.reduce_mean(x,axis=0)
        #print(mean)
        # 去中心化 防止数据过分逼近大的值 而忽略小的值
        x_new = x - mean
        #print(x_new)
        # 无偏差的协方差矩阵
        cov = tf.matmul(x_new,x_new,transpose_a=True)/(m - 1) 
        #print(cov)
        # 计算特征分解
        e,v = tf.linalg.eigh(cov,name="eigh")
        #print(e)
        #print(v)
        # 将特征值从大到小排序，选出前dim个的index
        e_index_sort = tf.math.top_k(e,sorted=True,k=dim)[1]
        #print(e_index_sort)
        # 提取前排序后dim个特征向量
        v_new = tf.gather(v,indices=e_index_sort)
        #print(v_new)
        # 降维操作
        pca = tf.matmul(x_new,v_new,transpose_b=True)
    return pca


# In[4]:


start = time.perf_counter()
pca_data = tf.constant(np.reshape(data.data,(data.data.shape[0],-1)),dtype=tf.float32)
pca_data1 = pca(pca_data,dim=2)
elapsed = (time.perf_counter()-start)
print(f'time use :{elapsed}')


# In[5]:


start = time.perf_counter()
pca_data = tf.constant(np.reshape(n,(n.shape[0],-1)),dtype=tf.float32)
pca_data = pca(pca_data,dim=2)
elapsed = (time.perf_counter()-start)
print(f'time use :{elapsed}')


# In[6]:


Y= data.target
pca = pca_data1.numpy()
plt.figure()
color=['red','green','blue']
for i, target_name in enumerate(data.target_names):
    plt.scatter(pca[Y==i,0],pca[Y==i,1],label = target_name, color = color[i])
plt.legend()
plt.title('pca')
plt.show


# In[ ]:




