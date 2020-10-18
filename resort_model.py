import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2
saver = tf.train.import_meta_graph('/Users/hycao/tensorflow_learn/path/to/model/model_test.ckpt.meta')
graph = tf.get_default_graph()
# saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "/Users/hycao/tensorflow_learn/path/to/model/model_test.ckpt")
    print(sess.run(result))
