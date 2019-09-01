import h5py
import numpy as np
import matplotlib.pyplot as plt


def load_cat_dataset():
    train_dataset = h5py.File('../steemit/data/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('../steemit/data/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    train_set_x_orig = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_orig = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_set_x = train_set_x_orig / 255
    test_set_x = test_set_x_orig / 255

    return train_set_x, train_set_y, test_set_x, test_set_y, classes


train_set_x, train_set_y, test_set_x, test_set_y, classes = load_cat_dataset()
print("train_set_x.shape", train_set_x.shape)
print("train_set_x.shape", train_set_y.shape)
print("train_set_x.shape", test_set_x.shape)
print("train_set_x.shape", test_set_y.shape)

# global w, b
np.random.seed(0)
# Initialize weights and bias
# There are 12288 weights and one bias value
# w = np.random.rand(train_set_x.shape[0])/100.0
w = np.zeros((train_set_x.shape[0]))
b = np.random.rand(1)

learning_rate = 0.001
iterations = 100

# Number of neurons of the layer
# Since this is the first layer, number of neurons equal to 12288
neurons_of_the_layer = train_set_x.shape[0]
# Size of the data set
m = train_set_x.shape[1]


# Here we consider sigmoid function as the activation function
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


# Calculate the cost for a single sample
# Represent the qqn (3)
def calculate_cost(a, y):
    return -y * np.log(a) - (1 - y) * np.log(1 - a)


# Calculate the eqn (1)
# It calculates the z wvalue for a single sample
def forward_propagation(w, b, x, n):
    z = b
    for i in range(0, n):
        z = z + w[i] * x[i]
    # Calculate the eqn (1)
    return sigmoid(z)


def back_propagation(dw, db, a, y, x):
    # Calculate eqn (6)
    for j in range(0, neurons_of_the_layer):
        dw[j] = dw[j] + (a - y) * x[j]
    # Calculate eqn (7)
    db = db + (a - train_set_y[0, i])
    return dw, db


def train():
    global w, b
    # At the beginning of the each iteration cost should be set as zero
    cost = 0
    # Initialize the derivative of w and b
    dw = np.zeros(neurons_of_the_layer)
    db = 0
    for i in range(0, m):
        a = forward_propagation(w, b, train_set_x[:, i], neurons_of_the_layer)
        # Related to eqn (3) and (4)
        c = calculate_cost(a, train_set_y[0, i])
        cost = cost + c
        # # Back propagation without a different method
        # for j in range(0, neurons_of_the_layer):
        #     dw[j] = dw[j] + (a - train_set_y[0, i]) * train_set_x[j, i]
        # db = db + (a - train_set_y[0, i])
        dw, db = back_propagation(dw, db, a, train_set_y[0, i], train_set_x[:, i])
    # This is equivalent to eqn (4)
    cost = cost / m
    dw = dw / m
    db = db / m

    # Update the weights and biases
    # calculate eqn (8) and (9)
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return cost


# Predict the values against the test dataset
def predict(x):
    y_predict = np.zeros((1, x.shape[1]))
    for i in range(0, x.shape[1]):
        a = forward_propagation(w, b, x[:, i], neurons_of_the_layer)
        if a > 0.5:
            y_predict[0, i] = 1
    return y_predict


costs = []
for i in range(0, iterations):
    cost = train()
    costs.append(cost)
    print("Cost for iteration " + str(i) + " is " + str(cost))

y_predict_train = predict(train_set_x)
y_predict_test = predict(test_set_x)

print("train accuracy: {} %".format(100 - np.mean(np.abs(y_predict_train - train_set_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(y_predict_test - test_set_y)) * 100))

plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Cost")
plt.show()
