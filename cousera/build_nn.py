import numpy as np
import matplotlib.pyplot as plt
from nn_layer import *

layers_dims = [12288, 20, 7, 5, 1]
np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
parameters = initialize_parameters_deep(layers_dims)

# Explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],
                                       -1).T  # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

# Initialize the dA values for each layer except last layer
d_activations = initialize_derivatives_of_activations(layers_dims, train_x.shape[1])


def run_nn():
    parameters['W1'], parameters['b1'], d_activations['dA0'], A1 = neural_network_layer(parameters['W1'],
                                                                                        parameters['b1'],
                                                                                        train_x, d_activations['dA1'],
                                                                                        learning_rate, 'relu')
    parameters['W2'], parameters['b2'], d_activations['dA1'], A2 = neural_network_layer(parameters['W2'],
                                                                                        parameters['b2'], A1,
                                                                                        d_activations['dA2'],
                                                                                        learning_rate,
                                                                                        'relu')
    parameters['W3'], parameters['b3'], d_activations['dA2'], A3 = neural_network_layer(parameters['W3'],
                                                                                        parameters['b3'], A2,
                                                                                        d_activations['dA3'],
                                                                                        learning_rate,
                                                                                        'relu')
    parameters['W4'], parameters['b4'], d_activations['dA3'], A4 = neural_network_layer(parameters['W4'],
                                                                                        parameters['b4'], A3,
                                                                                        d_activations['dA4'],
                                                                                        learning_rate,
                                                                                        'sigmoid')

    cost = compute_cost(A4, train_y)
    d_activations['dA' + str(len(layers_dims) - 1)] = - (np.divide(train_y, A4) - np.divide(1 - train_y, 1 - A4))
    return cost, A4


def predict(x, y):
    A1, Z1 = neural_network_layer(parameters['W1'], parameters['b1'], x, '', '', 'relu', True)
    A2, Z1 = neural_network_layer(parameters['W2'], parameters['b2'], A1, '', '', 'relu', True)
    A3, Z1 = neural_network_layer(parameters['W3'], parameters['b3'], A2, '', '', 'relu', True)
    A4, Z1 = neural_network_layer(parameters['W4'], parameters['b4'], A3, '', '', 'sigmoid', True)

    m = x.shape[1]
    p = np.zeros((1, m))

    for i in range(0, A4.shape[1]):
        if A4[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("Accuracy: " + str(np.sum((p == y) / m)))

    return p


learning_rate = 0.002
num_iterations = 8400
print_cost = True
costs = []

for i in range(0, num_iterations):
    cost, A4 = run_nn()
    # Print the cost every 100 training example
    if print_cost and i % 100 == 0:
        print("Cost after iteration %i: %f" % (i, cost))
    if print_cost and i % 100 == 0:
        costs.append(cost)

# plot the cost
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()

pred_train = predict(train_x, train_y)
pred_test = predict(test_x, test_y)
