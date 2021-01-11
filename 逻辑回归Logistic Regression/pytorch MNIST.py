#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 600
learning_rate = 0.001
use_gpu = torch.cuda.is_available()
print(use_gpu)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 下载数据
train_dataset = torchvision.datasets.MNIST(root='/data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='/data', 
                                          train=False, 
                                          transform=transforms.ToTensor())
# 设置每次运行的数据量
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# 建立模型
model = nn.Linear(input_size, num_classes)
# 设定损失函数（交叉熵损失函数）
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
# 开始训练

total_step = len(train_loader)
start = time.perf_counter()
if(use_gpu):
    model = model.cuda()
    criterion = criterion.cuda()
    train_loader = train_loader.to(device)
    test_loader = test_loader.to(device)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28)
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 梯度下降
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #if (i+1) % 300 == 0:
            #print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   #.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
elapsed = (time.perf_counter() - start)
print("Time used:",elapsed)


# In[8]:

