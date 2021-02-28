#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time


# In[2]:


train_dataset = torchvision.datasets.MNIST(root='/data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='/data', 
                                          train=False, 
                                          transform=transforms.ToTensor())


# In[25]:


#数据准备
x_train = train_dataset.data.float()#训练数据
y_train = train_dataset.targets#训练数据的标签
x_test = test_dataset.data.float()#测试数据
y_test = test_dataset.targets#测试数据的标签


# In[34]:


accuracy=0
number=1000
start = time.perf_counter()
for epoch in range(number):
    # 计算L2距离
    distance = torch.sqrt(torch.sum(torch.sum(torch.pow(x_train-x_test[epoch],2),dim=2),dim=1))
    # 获取最小距离的索引
    nn_index =  torch.argmin(distance, 0)
    #if epoch  ==0:
    #print("Test", epoch, "Prediction:", y_train[nn_index].item(),"True Class:", y_test[epoch].item())
    # 计算精确度
    if y_train[nn_index].item() == y_test[epoch].item():
        accuracy += 1/number
elapsed = (time.perf_counter() - start)#计算所用时间
print("Time used:",elapsed)#输出结果
print(f'accuracy:{accuracy}')


# In[ ]:




