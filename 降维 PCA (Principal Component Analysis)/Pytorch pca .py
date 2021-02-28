#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import time
from sklearn import datasets
import matplotlib.pyplot as plt


# In[2]:


data = datasets.load_iris(return_X_y=False)
n0=2*np.random.randn(5000,1000)+1
n1=-2*np.random.randn(5000,1000)-1
n=np.vstack((n0,n1))


# In[3]:


def pca(x,dim=2):
    m,n=x.shape[0],x.shape[1]
    #print(m,n)
    X=torch.from_numpy(x)
    #去中心化 防止数据过分逼近大的值 而忽略小的值
    X_mean = torch.mean(X,0)
    X=X-X_mean.expand_as(X)
    # 无偏差的协方差矩阵
    cov = torch.matmul(X.T,X)/(m - 1)
    #print(cov)
    # 计算特征分解
    e,v = torch.eig(cov,eigenvectors=True)
    #print(e)
    #print(v.shape[0])
    e=torch.mean(e,1)*2
    #print(e)
    # 将特征值从大到小排序，选出前dim个的index
    sorted, e_index_sort = torch.sort(e,descending=True)
    e_index_sort=torch.gather(e_index_sort,-1,torch.LongTensor([0,1]))
    v_new = torch.index_select(v, 0, e_index_sort)
    #print(v_new)
    # 降维操作
    pca = torch.matmul(X,v_new.T)
    return(pca)


# In[4]:


start = time.perf_counter()
pca_data1 = pca(data.data,dim=2)
elapsed = (time.perf_counter()-start)
print(f'time use :{elapsed}')


# In[5]:


start = time.perf_counter()
pca_data2 = pca(n,dim=2)
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




