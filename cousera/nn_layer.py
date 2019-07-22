import numpy as np
import h5py


def load_data():
    train_dataset = h5py.File('data/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('data/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def forward_prop(a, w, b, activation_function):
    """
    Implement the forward propagation.

    :param a: Output of the layer previous layer
    :param w: Weights
    :param b: Bias
    :param activation_function: Activation function
    :return: Output of this layer
    """

    z = w.dot(a) + b
    if activation_function == "sigmoid":
        return sigmoid(z), z
    elif activation_function == "relu":
        return relu(z), z


def backward_prop(z, dA, a_prev, w, activation_function):
    """
    Implement the backward propagation.

    :param z: Linear output
    :param dA: Derivative of the activated output
    :param a_prev: Previous activated output
    :param w: Weights
    :param activation_function: Activation function
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

    :param z: numpy array of any shape
    :return: output of sigmoid(z), same shape as Z
    """

    A = 1 / (1 + np.exp(-z))
    return A


def relu(z):
    """
    Implement the RELU function.

    :param z: Output of the linear layer, of any shape
    :return: Post-activation parameter, of the same shape as Z
    """

    A = np.maximum(0, z)
    assert (A.shape == z.shape)
    return A


def neural_network_layer(W, b, A_prev, dA, learning_rate, activation_function, forward_only=False):
    """
    Implement a common layer including forward propagation and backward propagation.

    :param W: Weights
    :param b: Bias
    :param A_prev: Output of the previous layer
    :param dA: Derivative the output
    :param learning_rate: Learning rate
    :param activation_function: Activation function used
    :param forward_only: Calculate only the forward propagation
    :return:
    W -- Weights
    b -- Bias
    A -- Output of this layer
    dA_prev -- Output of the derivative of this layer
    """

    # Forward propagation
    A, Z = forward_prop(A_prev, W, b, activation_function)
    if forward_only:
        return A, Z

    dW, db, dA_prev = backward_prop(Z, dA, A_prev, W, activation_function)

    W = W - learning_rate * dW
    b = b - learning_rate * db

    return W, b, dA_prev, A


def initialize_parameters_deep(layer_dims):
    """
    Initialize the weights and bias.

    :param layer_dims: python array (list) containing the dimensions of each layer in our network
    :return: python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
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
    """
    Initialize the derivates of the activations.

    :param layer_dims: Number of layers
    :param batch_size: Size of the batch
    :return: Derivatives of the activations
    """
    np.random.seed(1)
    d_activations = {}
    L = len(layer_dims)

    for l in range(0, L):
        d_activations['dA' + str(l)] = np.random.randn(layer_dims[l], batch_size)
    return d_activations


def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    :param AL: Probability vector corresponding to your label predictions, shape (1, number of examples)
    :param Y: True "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    :return: Cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost
