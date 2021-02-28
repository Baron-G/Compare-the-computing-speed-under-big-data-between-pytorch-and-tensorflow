#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn


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
#`print(x_train)


# In[3]:


#plt.plot(x_train[:,0],x_train[:,1],"ob")
#x = np.arange(-3,4) 
#p = 2*x
#plt.plot(x,p) 
#plt.show()


# In[4]:


X_train=torch.from_numpy(x_train)
#print(X_train.dtype)
X_target = torch.from_numpy(x_target)
inputs = torch.as_tensor(X_train,dtype=torch.float32)
targets = torch.as_tensor(X_target,dtype=torch.float32)


# In[5]:


#定义模型
model = nn.Linear(2,1)


# In[6]:


#LOSS= L2+C*max(0,1-pred)  hinge loss 
#classification_term = torch.mean(torch.maximum(torch.tensor((0)),1 - model(inputs)*targets))
#print(classification_term)
#Loss = L2+classification_term
#print(Loss)


# In[7]:


opt= torch.optim.SGD(model.parameters(),lr=0.01)#选择优化器 SGD为随机梯度下降函数


# In[8]:


px=[]
py=[]
for i in range(50000):
    L2=torch.matmul(model.weight,model.weight.T)
    #print(L2)
    classification_term = torch.mean(torch.maximum(torch.tensor((0.)),1.-model(inputs)*targets))
    #print(classification_term)
    Loss = L2+classification_term
    #print(Loss)
    Loss.backward()#梯度下降函数
    opt.step()#用opt.step()实现model中的w和b的改变
    opt.zero_grad()
    px.append(i)
    py.append(Loss.item())


# In[9]:


[[a1, a2]] = model.weight
#print(model.bias)
[b] = model.bias
#print(a1)
#print(a2)
#plt.cla()
#plt.plot(px,py,'r-',lw=1)


# In[10]:


plt.plot(x_train[:50,0],x_train[:50,1],"ob")
plt.plot(x_train[50:,0],x_train[50:,1],"or")
p = -(a2.item()/a1.item())*x_train[:,0]-b.item()/a1.item()
p2=p-1
p3=p+1
plt.plot(x_train[:,0],p)
plt.plot(x_train[:,0],p2)
plt.plot(x_train[:,0],p3)
plt.show()


# In[ ]:




