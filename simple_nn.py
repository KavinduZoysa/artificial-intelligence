import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import datasets

# Load data
iris = datasets.load_iris()

# Number of epochs
epochs = 10000

# Learning rate
lr = 0.01

W1 = np.random.rand(5, 4)  # Layer 1 parameters
W2 = np.random.rand(3, 5)  # Layer 2 parameters

B1 = np.random.rand(1, 5)
B2 = np.random.rand(1, 3)

OP = np.zeros((iris.data.shape[0], 3), dtype=float)

# def read_input():
#     return pd.read_csv("input/iris.csv").values[0:150, 0:4]


# doing the forward propagation
def forward_propagation(parameters, x):
    return np.dot(parameters, x)


# activation function
def sigmoid(s):
    return 1/(1+np.exp(-s))


# derivative of activation function
def sigmoid_prime(s):
    return s * (1 - s)


for i in range(0, iris.target.shape[0]):
    if iris.target[i] == 0:
        OP[i, 0] = 1
        # OP[i, 1] = -1
        # OP[i, 2] = -1
    elif iris.target[i] == 1:
        # OP[i, 0] = -1
        OP[i, 1] = 1
        # OP[i, 2] = -1
    elif iris.target[i] == 2:
        # OP[i, 0] = -1
        # OP[i, 1] = -1
        OP[i, 2] = 1

# for i in range(0, iris.data.shape[0] - 1):
#     print(i)
#     x0 = iris.data[i, :]
#     x1 = forward_propagation(W1, x0) + B1
#     X1 = sigmoid(x1)
#     x2 = forward_propagation(W2, X1.T) + B2.T
#     X2 = sigmoid(x2)
#
#     delta2 = (X2.T - OP[i]).T*(X2*(1 - X2))
#     dW2 = np.dot(delta2, X1)
#     dB2 = delta2
#
#     delta1 = np.dot(W2.T, delta2)*(X1*(1 - X1)).T
#     dW1 = np.dot(delta1, np.array([x0]))
#     dB1 = delta1
#
#     W2 = W2 - 0.5*dW2
#     B2 = B2 - 0.5*dB2.T
#
#     W1 = W1 - dW1
#     B1 = B1 - dB1.T
#     print(W2)

for e in range(0, epochs):
    print("epoch=", e)
    for i in range(0, iris.data.shape[0] - 1):
        x0 = iris.data[i, :]
        x1 = forward_propagation(W1, x0) + B1
        X1 = sigmoid(x1)
        x2 = forward_propagation(W2, X1.T) + B2.T
        X2 = sigmoid(x2)

        delta2 = (X2.T - OP[i]).T * (X2 * (1 - X2))
        dW2 = np.dot(delta2, X1)
        dB2 = delta2

        delta1 = np.dot(W2.T, delta2) * (X1 * (1 - X1)).T
        dW1 = np.dot(delta1, np.array([x0]))
        dB1 = delta1

        W2 = W2 - lr * dW2
        B2 = B2 - lr * dB2.T

        W1 = W1 - lr * dW1
        B1 = B1 - lr * dB1.T


for j in range(0, iris.data.shape[0] - 1):
    x0 = iris.data[j, :]
    xt1 = forward_propagation(W1, x0) + B1
    Xt1 = sigmoid(xt1)
    xt2 = forward_propagation(W2, Xt1.T) + B2.T
    Xt2 = sigmoid(xt2)

    # print((OP[j] - Xt2.T))
    print("expected_output=", OP[j])
    print("actual_output", Xt2.T)




