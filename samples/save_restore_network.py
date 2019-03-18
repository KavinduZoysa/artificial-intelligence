import tensorflow as tf
from tensorflow.python import debug as tf_debug


def save_network():
    # Prepare to feed input, i.e. feed_dict and placeholders
    w1 = tf.placeholder("float", name="w1")
    w2 = tf.placeholder("float", name="w2")
    b1 = tf.Variable(3.0, name="bias")

    # Define a test operation that we will restore
    w3 = tf.add(w1, w2)
    w4 = tf.multiply(w3, b1, name="op_to_restore")
    with tf.Session() as sess1:
        sess1.run(tf.global_variables_initializer())
        # sess1 = tf_debug.LocalCLIDebugWrapperSession(sess1)
        saver = tf.train.Saver()
        print(sess1.run(w4, feed_dict={w1: 4, w2: 8}))
        saver.save(sess1, 'network/my_test_model')


def restore_network():
    with tf.Session() as sess2:
        saver = tf.train.import_meta_graph('/home/kavindu/workspace/PycharmProjects/ai/samples/network/my_test_model.meta')
        saver.restore(sess2, tf.train.latest_checkpoint('/home/kavindu/workspace/PycharmProjects/ai/samples/network'))
        # sess2 = tf_debug.LocalCLIDebugWrapperSession(sess2)
        graph = tf.get_default_graph()
        w1 = graph.get_tensor_by_name("w1:0")
        w2 = graph.get_tensor_by_name("w2:0")
        # feed_dict = {w1: 13.0, w2: 17.0}
        print(sess2.run(graph.get_tensor_by_name("op_to_restore:0"), feed_dict={w1: 4, w2: 8}))


save_network()
restore_network()
