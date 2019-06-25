import numpy as np
from sklearn import datasets

# X = [x1, x2].T
# W = [w1, w2]
# b
# z = w.T.X + b
# y_hat = a = sigma(z)
# L = -(y*log(a) + (1-y)*log(1-a))

# dL/da = -y/a + (1-y)/(1-a)
# da/dz = a(1-a)
# dL/dz = a-y

iris = datasets.load_iris()

low = 0
X_transpose = iris.data[low:100]
X = X_transpose.transpose()
Y = iris.target[low:100]


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


# Initialize the weights(w1, w2) and bias(b)
w1 = 0
w2 = 0
w3 = 0
w4 = 0
b = 0

# Initialize the loss
W = np.zeros((4, 1))
m = X.shape[1]

for j in range(0, 1000):
    # Initialize the derivatives(dJ/dw1, dJ/dw2, dJ/db)
    dw1 = 0
    dw2 = 0
    dw3 = 0
    dw4 = 0
    db = 0

    J = 0

    W[0] = w1
    W[1] = w2
    W[2] = w3
    W[3] = w3
    # Iterate over the input data
    for i in range(0, m):
        z = np.matmul(W.transpose(), X[:, i]) + b
        a = sigmoid(z)
        J = J + (-1) * (Y[i] * np.log(a) + (1 - Y[i]) * np.log(1 - a))
        # L = L - (Y[0, i] * np.log(a) + (1 - Y[0, i]) * np.log(1 - a))
        dw1 = dw1 + X[0, i] * (a - Y[i])
        dw2 = dw2 + X[1, i] * (a - Y[i])
        dw3 = dw3 + X[2, i] * (a - Y[i])
        dw4 = dw4 + X[3, i] * (a - Y[i])
        db = db + a - Y[i]
    # Get the average value
    J = J / m
    dw1 = dw1 / m
    dw2 = dw2 / m
    dw3 = dw3 / m
    dw4 = dw4 / m
    db = db / m

    # Update the weights
    w1 = w1 - 0.01 * dw1
    w2 = w2 - 0.01 * dw2
    w3 = w3 - 0.01 * dw3
    w4 = w4 - 0.01 * dw4
    b = b - 0.01 * db

# Test the output
for i in range(0, m):
    z = np.matmul(W.transpose(), X[:, i]) + b
    a = sigmoid(z)
    if a > 0.5:
        a = 1
    else:
        a = 0
    print(a - Y[i])
