
import tensorflow as tf
import numpy as np
import time

tf.compat.v1.disable_eager_execution()
n0=2*np.random.randn(5000,1000)+1
n1=-2*np.random.randn(5000,1000)+1
n = np.vstack((n0,n1))
m0 = np.zeros((5000,1))
m1 = np.ones((5000,1))
m= np.vstack((m0,m1))
#print(m0.dtype)


# In[6]:


N = tf.Variable(tf.constant(n,dtype=tf.float32))#创建一个tensor常量
M = tf.Variable(tf.constant(m,dtype=tf.float32))
w = tf.Variable(tf.random.normal([1000,1]))
b = tf.Variable(tf.random.normal([10000,1]))


# In[7]:


#用softmax构建逻辑回归模型
y_pred = tf.nn.sigmoid(tf.matmul(N,w) + b)
#损失函数（交叉熵）
cost = tf.reduce_mean(-tf.reduce_sum(M*tf.compat.v1.log(tf.clip_by_value(y_pred,1e-8,1.0)),1))
# 梯度下降
optimizer = tf.compat.v1.train.GradientDescentOptimizer(1e-3).minimize(cost)
#初始变量
init = tf.compat.v1.global_variables_initializer()


# In[8]:


num_epoch=1000
start = time.perf_counter()
with tf.compat.v1.Session() as sess:
    with tf.device('/gpu:0'):
    #初始化所有变量
        sess.run(init)
     #开始训练
        for epoch in range(num_epoch):
            sess.run(optimizer)
            train_cost=sess.run(cost)
            #cost_accum.append(train_cost)
            #if (epoch + 1) % 50 == 0:
                #print('epoch {}'.format(epoch+1)) # 训练轮数
                #print('loss is {:.4f}'.format(train_cost))  # 输出误差
elapsed = (time.perf_counter() - start)
print("Time used:",elapsed)


# In[ ]:




