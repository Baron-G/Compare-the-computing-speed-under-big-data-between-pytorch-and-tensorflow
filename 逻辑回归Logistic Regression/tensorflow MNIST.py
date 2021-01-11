
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import time

# MNIST数据集参数
num_classes = 10  # 数字0到9, 10类
num_features = 784  # 28*28

# 训练参数
learning_rate = 0.01
training_steps = 1000
batch_size = 256
display_step =50

# 预处理数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 转为float32
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# 转为一维向量
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
# [0, 255] 到 [0, 1]
x_train, x_test = x_train / 255, x_test / 255

# tf.data.Dataset.from_tensor_slices 是使用x_train, y_train构建数据集
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# 将数据集打乱，并设置batch_size大小
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# 权重[748, 10]，图片大小28*28，类数
W = tf.Variable(tf.ones([num_features, num_classes]), name="weight")
# 偏置[10]，共10类
b = tf.Variable(tf.zeros([num_classes]), name="bias")

# 逻辑回归函数
def logistic_regression(x):
    return tf.nn.softmax(tf.matmul(x, W) + b)

# 损失函数
def cross_entropy(y_pred, y_true):
    # tf.one_hot()函数的作用是将一个值化为一个概率分布的向量
    y_true = tf.one_hot(y_true, depth=num_classes)
    # tf.clip_by_value将y_pred的值控制在1e-9和1.0之间
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.0)
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

# 计算精度
def accuracy(y_pred, y_true):
    # tf.cast作用是类型转换
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 优化器采用随机梯度下降
optimizer = tf.optimizers.SGD(learning_rate)

# 梯度下降
def run_optimization(x, y):
    with tf.GradientTape() as g:
        pred = logistic_regression(x)
        loss = cross_entropy(pred, y)
    # 计算梯度
    gradients = g.gradient(loss, [W, b])
    # 更新梯度
    optimizer.apply_gradients(zip(gradients, [W, b]))

# 开始训练
start = time.perf_counter()
for epoch in range(5):
    for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
        run_optimization(batch_x, batch_y)
        #if step % display_step == 0:
            #pred = logistic_regression(batch_x)
            #loss = cross_entropy(pred, batch_y)
            #acc = accuracy(pred, batch_y)
            #print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
    
# 测试模型的准确率
#pred = logistic_regression(x_test)
#print("Test Accuracy: %f" % accuracy(pred, y_test))
elapsed = (time.perf_counter() - start)
print("Time used:",elapsed)


# In[ ]:





