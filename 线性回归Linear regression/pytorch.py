#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

use_cuda = torch.cuda.is_available()
x = np.ones((10000,1000))
inputs = x+np.random.rand(10000,1000)
targets = np.ones((10000,1))
#inputs = np.random.normal(0,1,(15,3))
#targets = np.random.normal(0,1,(15,1))
#print(inputs)
#print(targets)


# In[2]:


#将numpy转化为tensor 并设为float32类型
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
inputs = torch.as_tensor(inputs,dtype=torch.float32)
targets = torch.as_tensor(targets,dtype=torch.float32)
#print(inputs)
#print(targets)


# In[3]:


model = nn.Linear(1000,1)#创建权重和偏置的初始化矩阵
#print(model.weight)
#print(model.bias)
#list(model.parameters())


# In[4]:


if use_cuda:
    inputs=inputs.cuda()
    targets = targets.cuda()
    model=model.cuda()
from torch.utils.data import TensorDataset#创建一个TensorDataset，可以读取 inputs 和 targets 的行作为元组
train_ds=TensorDataset(inputs,targets)#将inputs,targets放入train_ds中
#print(train_ds[:3])


# In[5]:


#from torch.utils.data import DataLoader#创建一个 DataLoader，它可以在训练时将数据分成预定义大小的批次。它还能提供其它效用程序，如数据的混洗和随机采样
#batch_size=1000#设定一次运行的数据量
#train_dl=DataLoader(train_ds,batch_size,shuffle=True)#在每次迭代中，数据加载器都会返回一批给定批大小的数据。如果 shuffle 设为 True，则在创建批之前会对训练数据进行混洗。混洗能帮助优化算法的输入随机化，这能实现损失的更快下降。
#for xb,yb in train_dl:
#    print(xb)
#    print(yb)
#    break


# In[6]:


preds = model(inputs)#得到模拟值但是和真实值之相差较大
#print(preds)


# In[7]:


loss_fn = nn.MSELoss()#定义损失函数 预测值和真实值差值平方的平均数
loss = loss_fn(preds,targets)
#print(loss)


# In[8]:


opt= torch.optim.SGD(model.parameters(),lr=0.0001)#选择优化器 SGD为随机梯度下降函数


# In[9]:


def fit(num_epochs,model,loss_fn,opt):
    px,py=[],[]
    for epoch in range(num_epochs):
            pred=model(inputs)
            #print(model.weight)
            #print(model.bias)
            loss = loss_fn(pred,targets)
            loss.backward()
            opt.step()#用opt.step()实现model中的w和b的改变
            opt.zero_grad()
            px.append(epoch)
            py.append(loss.item())
            if(epoch+1)%10==0:
                print("Epoch[{}/{}],Loss:{:.4f}".format(epoch+1,num_epochs,loss.item()))
            if epoch % 10 ==0:
                plt.cla()
                plt.plot(px,py,'r-',lw=1)
start = time.perf_counter()
fit(100,model,loss_fn,opt)
elapsed = (time.perf_counter() - start)
print("Time used:",elapsed)

