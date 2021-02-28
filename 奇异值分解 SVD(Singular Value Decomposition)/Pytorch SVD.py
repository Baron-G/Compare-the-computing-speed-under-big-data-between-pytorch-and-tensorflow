#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import torch
import numpy as np
import matplotlib.pyplot as plt
import time


# In[2]:


def pca_svd(data,k=2):
    #将数据转化为tensor
    X=torch.from_numpy(data)
    #print(X)
    #去中心化 避免数据过度损失
    X_mean= torch.mean(X,0)
    X=X-X_mean.expand_as(X)
    #print(X)
    #svd计算
    U,S,V = torch.svd(X)
    #print(S)
    #print(V[:,:k])
    return torch.mm(X,V[:,:k])


# In[3]:


#数据准备
iris = datasets.load_iris()
X= iris.data
Y= iris.target
start = time.perf_counter()
X_PCA = pca_svd(X)
elapsed = (time.perf_counter()-start)
print(f'time use :{elapsed}')
pca = X_PCA.numpy()
#print(pca)


# In[4]:


plt.figure()
color=['red','green','blue']
for i, target_name in enumerate(iris.target_names):
    plt.scatter(pca[Y==i,0],pca[Y==i,1],label = target_name, color = color[i])
plt.legend()
plt.title('svd')
plt.show


# In[5]:


n0=2*np.random.randn(5000,1000)+1
n1=-2*np.random.randn(5000,1000)-1
n=np.vstack((n0,n1))
start = time.perf_counter()
pca_data = pca_svd(n,k=2)
elapsed = (time.perf_counter()-start)
print(f'time use :{elapsed}')


# In[ ]:




