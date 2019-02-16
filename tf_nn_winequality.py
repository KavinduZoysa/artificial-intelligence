import tensorflow as tf
import numpy as np
import csv

reader = csv.reader(open("/home/kavindu/workspace/PycharmProjects/ai/input/winequality-white.csv", "r"), delimiter=",")
train_data = list(reader)
train_data = np.array(train_data)

# Initialize placeholders
# Number of rows are not defined
# Number of columns = 11
x = tf.placeholder(dtype=tf.float32, shape=[None, 11])
# Number of columns = None
y = tf.placeholder(dtype=tf.int32, shape=[None, 1])

# First hidden layer
# Number of neurons = 15
layer1 = tf.contrib.layers.fully_connected(x, 15, tf.nn.relu)

# Output layer
output = tf.contrib.layers.fully_connected(layer1, 1)

# Define loss
loss = tf.losses.mean_squared_error(labels=y, predictions=output)

# Define an optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10101):
        print('EPOCH', i)
        accuracy_val, loss_val = sess.run([train_op, loss], feed_dict={x: train_data[:, 0:11], y: train_data[:, 11:12]})
        if i % 10 == 0:
            print("Loss: ", loss_val)
        print('DONE WITH EPOCH')
    predicted = sess.run([output], feed_dict={x: train_data[:, 0:11]})
    sess.close()

print(predicted)