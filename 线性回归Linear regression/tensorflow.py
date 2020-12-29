import numpy as np
import tensorflow as tf

px,py=[],[]
x = np.ones((10000,1000))
inputs = x+np.random.rand(10000,1000)
targets = np.ones((10000,1))

X = tf.constant(inputs,dtype=tf.float32)#创建一个tensor常量
y = tf.constant(targets,dtype=tf.float32)
#print(X)
a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]
#print(a.dtype)
#print(b.dtype)
#print(X.dtype)
num_epoch = 100
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)        
# 定义随机梯度下降法中的优化器和学习率
start = time.perf_counter()
for e in range(num_epoch):
    with tf.device('/gpu:0'):
        # 使用tf.GradientTape()记录损失函数的梯度信息
        with tf.GradientTape() as tape:
            y_pred = a * X + b
            loss = tf.reduce_mean(tf.square(y_pred - y))#对每一个数平方并累加
            px.append(e)
            py.append(loss)
        # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
        grads = tape.gradient(loss, variables)
        # TensorFlow自动根据梯度更新参数
        optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
        if num_epoch % 10 ==0:
            plt.cla()
            plt.plot(px,py,'r-',lw=1) 
print(loss)
elapsed = (time.perf_counter() - start)
print("Time used:",elapsed)






