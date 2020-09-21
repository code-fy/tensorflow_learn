import tensorflow as tf
import numpy as np

tf.estimator
tf.compat.v1.disable_eager_execution()
# 定义数据batch的大小
batch_size = 8
# 定义神经网络的参数
w1 = tf.Variable(tf.random.normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random.normal([3, 1], stddev=1, seed=1))
# 定义输入节点和输出节点
x = tf.compat.v1.placeholder(tf.float32, shape=(None, 2), name='x-put')
y_ = tf.compat.v1.placeholder(tf.float32, shape=(None, 2), name='y-put')
# 定义神经网络前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# 定义损失函数和反向传播算法
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(
    y_ * tf.math.log(tf.clip_by_value(y, 1e-10, 1.0))
    + (1-y_) * tf.math.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
train_step = tf.compat.v1.train.GradientDescentOptimizer.minimize(cross_entropy)
# 生成随机模拟数据集
rdm = np.random.RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 定义规则给出样本标签
Y = [[int(x1 + x2 < 1) for (x1, x2) in X]]

with tf.compat.v1.Session() as sess:
    init_op = tf.compat.v1.global_variables_initializer
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))

    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(
            train_step, feed_dict={x: X[start:end], y:Y[start:end]}
        )
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
    print(sess.run(w1))
    print(sess.run(w2))
print("怎么回事")