import numpy as np


def forward_prop(a, w, b, activation_function):
    """
    Implement the forward propagation.

    :argument:
    a -- Output of the layer previous layer
    w -- Weights
    b -- Bias
    activation_function -- Activation function

    :return:
    Output of this layer
    """

    z = w.dot(a) + b
    if activation_function == "sigmoid":
        return sigmoid(z), z
    elif activation_function == "relu":
        return relu(z), z


def backward_prop(z, dA, a_prev, w, activation_function):
    """
    Implement the backward propagation.

    :argument:
    dA -- Derivative of the activated output
    activation_function -- Activation function
    z -- Linear output
    a_prev -- Previous activated output
    w -- Weights
    :return:
    dW -- derivative of weights
    db -- derivative of bias
    dA_prev -- Derivative of the previous activated output
    """

    dz = 0
    if activation_function == "sigmoid":
        s = 1 / (1 + np.exp(-z))
        dz = dA * s * (1 - s)

        assert (dz.shape == z.shape)

    elif activation_function == "relu":
        dz = np.array(dA, copy=True)
        dz[z <= 0] = 0

        assert (dz.shape == z.shape)

    m = a_prev.shape[1]
    dW = 1. / m * np.dot(dz, a_prev.T)
    db = 1. / m * np.sum(dz, axis=1, keepdims=True)
    dA_prev = np.dot(w.T, dz)
    return dW, db, dA_prev


def sigmoid(z):
    """
    Implements the sigmoid activation in numpy

    :argument:
    z -- numpy array of any shape

    :return:
    A -- output of sigmoid(z), same shape as Z
    """

    A = 1 / (1 + np.exp(-z))
    return A


def relu(z):
    """
    Implement the RELU function.

    :argument:
    Z -- Output of the linear layer, of any shape

    :return:
    A -- Post-activation parameter, of the same shape as Z
    """

    A = np.maximum(0, z)
    assert (A.shape == z.shape)
    return A


def neural_network_layer(W, b, A_prev, dA, learning_rate, activation_function):
    """
    Implement a common layer including forward propagation and backward propagation.

    :argument:
    W -- Weights
    b -- Bias
    A_prev -- Output of the previous layer
    dA -- Derivative the output
    learning_rate -- Learning rate
    activation_function -- Activation function used

    :return:
    W -- Weights
    b -- Bias
    A -- Output of this layer
    dA_prev -- Output of the derivative of this layer
    """

    # Forward propagation
    A, Z = forward_prop(A_prev, W, b, activation_function)
    dW, db, dA_prev = backward_prop(Z, dA, A_prev, W, activation_function)

    W = W - learning_rate * dW
    b = b - learning_rate * db

    return W, b, dA_prev, A


def initialize_parameters_deep(layer_dims):
    """
    :argument:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    :return:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(
            layer_dims[l - 1])  # *0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def initialize_derivatives_of_activations(layer_dims, batch_size):
    np.random.seed(1)
    d_activations = {}
    L = len(layer_dims)

    for l in range(0, L):
        d_activations['dA' + str(l)] = np.random.randn(layer_dims[l], batch_size)
    return d_activations


def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost



