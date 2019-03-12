import tensorflow as tf
import numpy as np
import csv

reader = csv.reader(open("/home/kavindu/workspace/PycharmProjects/ai/input/winequality-white.csv", "r"), delimiter=",")
train_data = list(reader)
train_data = np.array(train_data)

# Initialize placeholders
# Number of rows are not defined
# Number of columns = 11
x = tf.placeholder(dtype=tf.float32, shape=[None, 11], name='xx')
# Number of columns = None
# Output is a one-hot vector
y = tf.placeholder(dtype=tf.int32, shape=[None, 10], name='yy')

# First hidden layer
# Number of neurons = 20
layer1 = tf.contrib.layers.fully_connected(x, 40, tf.nn.relu)

# Second hidden layer
# Number of neurons = 10
layer2 = tf.contrib.layers.fully_connected(layer1, 25, tf.nn.relu)

# Third hidden layer
# Number of neurons = 10
output = tf.contrib.layers.fully_connected(layer2, 10)

# Define a loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))

# Define an optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    flatten_output = sess.run(tf.reshape(train_data[:, 11:12], [-1]))
    one_hot_output = sess.run(tf.one_hot(flatten_output, 10))

    for i in range(1000):
        print('EPOCH', i)
        accuracy_val, loss_val = sess.run([train_op, loss], feed_dict={x: train_data[:, 0:11], y: one_hot_output})
        if i % 10 == 0:
            print("Loss: ", loss_val)
        print('DONE WITH EPOCH')

    ss = sess.run([output], feed_dict={x: train_data[1:2, 0:11]})
    print(ss)
