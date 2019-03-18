import tensorflow as tf


# Save the model
def save_model():
    w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
    w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        print(sess.run(w2))
        saver.save(sess, 'models/my_test_models')


# Restore the model
def restore_model():
    saver = tf.train.import_meta_graph('/home/kavindu/workspace/PycharmProjects/ai/samples/models/my_test_models.meta')
    with tf.Session() as sess:
        sess = tf.Session()
        saver.restore(sess, tf.train.latest_checkpoint('/home/kavindu/workspace/PycharmProjects/ai/samples/models'))
        print(sess.run('w2:0'))


save_model()
restore_model()
