from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("./MINIST_data/", one_hot=True)
print("Train data size :", mnist.train.num_examples)
print("Test data size: ", mnist.test.num_examples)
print("Example training data: ", mnist.train.images[0])
print("Example training data label: ", mnist.train.labels[0])
batch_size = 100
xs, xy = mnist.train.next_batch(batch_size)
print("X shape:", xs.shape)
print("Y shape:", xy.shape)
# 输入的神经元个数
INPUT_NODE = 784
# 输出的神经元个数
OUTPUT_NODE = 10
# 隐藏层神经元个数
LAYER1_NODE = 500
# 梯度下降的batch，越小越接近随机梯度下降，越大越接近梯度下降
BATCH_SIZE = 100
# 基础学习率
LEARNING_RATE_BASE = 0.1
# 学习率衰减系数
LEARNING_RATE_DECAY = 0.99
# 正则化系数
REGULAIZATION_RATE = 0.001
# 迭代次数
TRAINING_STEPS = 8000
# 滑动平均衰减率
MOVING_AVGRAGE_DECAY = 0.99

def interence1(input_tensor, reuse=False):
    with tf.variable_scope('layer1', reuse=reuse):
        weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE],initializer=tf.truncated_normal(stddev=0.1))
        biase = tf.get_variable("biases", [LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights) + biase)

    with tf.variable_scope('layer2', reuse=reuse):
        weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE],initializer=tf.truncated_normal(stddev=0.1))
        biase = tf.get_variable("biases", [LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1,weights) + biase
    return layer2

def inference(input_tensor, avg_class, weight1, biases1, weight2, biases2):
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weight1) + biases1)
        return tf.matmul(layer1, weight2) + biases2

    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weight1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weight2)) + avg_class.average(biases2)

def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")
    # 生成隐藏层参数
    weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biase1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weight2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biase2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    y = inference(x, None, weight1, biase1, weight2, biase2)
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVGRAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weight1, biase1, weight2, biase2)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularizer = tf.contrib.layers.l2_regularizer(REGULAIZATION_RATE)
    regularization = regularizer(weight1) + regularizer(weight2)
    loss = cross_entropy_mean + regularization
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name="train")

    correct_prediction = tf.equal(tf.arg_max(average_y, 1), tf.arg_max(y_, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(acc, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy" "using average model is %g " % (i, validate_acc))
            xs, xy = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: xy})

        test_acc = sess.run(acc, feed_dict=test_feed)
        print("After %d training step(s), validation accuracy" "using average model is %g " % (TRAINING_STEPS, test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets("./MINIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
    print("over train")


