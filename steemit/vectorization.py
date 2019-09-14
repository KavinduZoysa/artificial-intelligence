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
# Number of neurons of the layer
# Since this is the first layer, number of neurons equal to 12288
neurons_of_the_layer = train_set_x.shape[0]
# Initialize weights and bias
# There are 12288 weights and one bias value
# w = np.random.rand(train_set_x.shape[0])/100.0
w = np.zeros(neurons_of_the_layer)
b = np.random.rand(1)

learning_rate = 0.005
iterations = 2000

# Size of the data set
m = train_set_x.shape[1]


# Here we consider sigmoid function as the activation function
def sigmoid(X):
    S = 1 / (1 + np.exp(-X))
    return S


# Calculate the cost for a single sample
# Represent the eqn (12)
def calculate_cost(A, Y):
    return np.sum(-Y * np.log(A) - (1 - Y) * np.log(1 - A)) / m


# Calculate the eqn (10)
# It calculates the z value for a single sample
def forward_propagation(W, B, X):
    Z = np.dot(W, X) + B
    # Calculate the eqn (11)
    return sigmoid(Z)


def back_propagation(A, Y, X):
    # Calculate the eqn (13)
    dW = np.dot((A - Y), X.T) / m
    # Calculate the eqn (14)
    dB = np.sum(A - Y) / m
    return dW, dB


def train(W, B):
    A = forward_propagation(W, B, train_set_x)
    cost = calculate_cost(A, train_set_y)
    dW, dB = back_propagation(A, train_set_y, train_set_x)
    return dW, dB, cost


def predict(w, b, X):
    Y_pred = np.zeros((1, X.shape[1]))

    A = forward_propagation(w, b, X)

    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_pred[0, i] = 1

    return Y_pred


costs = []
for i in range(0, iterations):
    dw, db, c = train(w, b)
    costs.append(c)
    # Calculate the eqn (15)
    w = w - learning_rate * dw
    # Calculate the eqn (16)
    b = b - learning_rate * db

y_predict_train = predict(w, b, train_set_x)
y_predict_test = predict(w, b, test_set_x)

print("train accuracy: {} %".format(100 - np.mean(np.abs(y_predict_train - train_set_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(y_predict_test - test_set_y)) * 100))

plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Cost")
plt.show()
