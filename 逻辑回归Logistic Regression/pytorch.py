#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.autograd import Variable
import torch.nn as nn
import time
import matplotlib.pyplot as plt
##创建数据
n_data = torch.ones(5000,1000) 
#创建x0，x1内数据的正太分布
x0 = torch.normal(2*n_data, 1)
#print(x0)
x1 = torch.normal(-2*n_data,1)
# orch.cat 是在合并数据
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y0 = torch.zeros(5000,1)
y1 = torch.ones(5000,1)
y = torch.cat((y0, y1), 0).type(torch.FloatTensor)# FloatTensor = 32-bit floating
print(x.shape)


# In[2]:


#定义模型
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()#继承父类的所有属性
        self.lr = nn.Linear(1000, 1)#建立初始的权重和偏置
        self.sm = nn.Sigmoid()
 
    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x


# In[3]:


logistic_model = LogisticRegression()

loss = nn.BCELoss()#导入最大然概率模型
#选择随机梯度下降优化器
optimizer = torch.optim.SGD(logistic_model.parameters(), lr=1e-3, momentum=0.9)
#if torch.cuda.is_available():#如果有GPU则将Tensor放到GPU中进行运算
       # x_data = Variable(x.cuda())
       # y_data = Variable(y.cuda())
        #logistic_model.cuda()
#else:
x_data = Variable(x)
y_data = Variable(y)


# In[4]:


#开始训练
start = time.perf_counter()
for epoch in range(1000):
    out = logistic_model(x_data)
    Loss = loss(out, y_data)
    mask = out.ge(0.5).float()  # 以0.5为阈值进行分类
    correct = (mask == y_data).sum()  # 计算正确预测的样本个数
    acc = correct.item() / x_data.size(0)  # 计算精度
    Loss.backward()
    optimizer.zero_grad()#将梯度清零
    optimizer.step()#更新权重和偏置
    # 每隔20轮打印一下当前的误差和精度
    #if (epoch + 1) % 100 == 0:
        #print('epoch {}'.format(epoch+1)) # 训练轮数
        #print('loss is {:.4f}'.format(Loss.data.item()))  # 输出误差
        #print('acc is {:.4f}'.format(acc))  # 精度
    #if epoch % 10 ==0:
                #plt.cla()
                #plt.plot(px,py,'r-',lw=1)    
elapsed = (time.perf_counter() - start)
print("Time used:",elapsed)







