import tensorflow as tf
import numpy as np


# number of neurons in each layer
input_num_units = 28*28
hidden_num_units = 500
output_num_units = 10

# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train/255.0, x_test/255.0


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    temp_batch = unclean_batch_x / unclean_batch_x.max()

    return temp_batch


def batch_creator(batch_size, dataset_length, dataset_name):
    # batch_size = 128
    # dataset_length = 6000
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)

    batch_x = eval('x_' + dataset_name)[[batch_mask]].reshape(-1, input_num_units)
    batch_x = preproc(batch_x)

    if dataset_name == 'train':
        batch_y = eval('y_' + dataset_name)[[batch_mask]]
        batch_y = dense_to_one_hot(batch_y)

    return batch_x, batch_y


# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])


# set remaining variables
epochs = 5
batch_size = 128
learning_rate = 0.01


weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
}


biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}


hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)

output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # create initialized variables
    sess.run(init)
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(x_train.shape[0] / batch_size)
        for i in range(total_batch):
            batch_x, batch_y = batch_creator(batch_size, x_train.shape[0], 'train')
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            print("ok")
        print("ok")
