"""
Multi-layer perceptron
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
%matplotlib qt

"""
initialize the weights matrice with random values on a normal distribution
"""
def init_weights(x, layers):
    weights = []
    weights.append(np.random.normal(0,0.1, (layers[0], len(x))))


    for layer in range(1, len(layers)):
        weights.append(np.random.normal(0,0.1, (layers[layer], layers[layer -1])))

    return (weights)


#Sigmoid function
def sigmoid(x):
    sig = 1 / (1+ np.exp(-x))

    return (sig)

#Network error
def net_error(tar, out):
    err = 0.5 * np.power(tar - out, 2)

    return (err)


"""
#feed forward

x: input vector
layers: [nb_node_layer_1, nb_node_layer_2, ...]
weights: [[w_layer_1:[w_node_1],[w_node_2]], [w_layer_2:[w_node_1],[w_node_2]]]
bias: [b_layer_1, b_layer_2]
"""
def feed_forward(x, layers, weights, bias):
    nb_layers = len(layers)
    a= []

    z = np.dot(x, weights[0].T) + bias[0]
    a.append(sigmoid(z))

    for layer in range(1, len(layers)):
        multiply = np.dot(a[layer -1], weights[layer].T)
        z =  multiply + bias[layer]
        a.append(np.array(sigmoid(z)))
    return a

#Back-propagation
def back_propagation(x, tar, weights, bias, layers):

    n_layers = len(layers)
    # output layer error
    L_layer = layers[n_layers - 1]

    #L_error value is (0.5(sigmoid(s_1) - target)^2)'
    # the derivative value: (1-sigmoid^2(s))(sigmoid(s) - tar)
    # -(tar - a_L)*a_L*(1-a_L)
    L_error = -(tar - L_layer)*L_layer*(1 - L_layer)


    # hidden layers errors
    # l_errors[layer -1] = l_sum*layers[layer -1 ]*(1- layers[layer -1])
    l_errors = [0 for i in range(n_layers)]
    l_errors[n_layers -1] = L_error

    for layer in range(n_layers - 1, 0, -1):
        l_weights = weights[layer]

        #sumarize
        l_sum = []
        for weight in l_weights.T:
             l_sum.append(np.sum(weight*l_errors[layer]))
        l_sum = np.array(l_sum)


        # compute the previous l_error vector
        l_errors[layer -1] = l_sum*layers[layer -1 ]*(1- layers[layer -1])

    # New weights and bias
    my_layers = copy.deepcopy(layers)
    my_layers.insert(0, x)

    n_weights = [0]*(len(my_layers) -1 )
    n_bias = [0]*(len(my_layers) - 1)
    for i in range(len(n_weights)):
        non_reshaped_l_errors = l_errors[i]
        l_errors[i] = np.reshape(l_errors[i], (len(l_errors[i]), 1))
        my_layers[i] = np.reshape(my_layers[i],(len(my_layers[i]),1))
        n_weights[i] = weights[i] - alpha*np.multiply(l_errors[i], my_layers[i].T)
        n_bias[i] = bias[i] - alpha*non_reshaped_l_errors

    return (n_weights, n_bias)


#################
# MAIN PART
#################

X = np.array([[0,0], [0,1], [1,0], [1,1]])
tar = np.array([0,1,1,0])

layers = [2, 1]
alpha = 0.5
epochs = 8000

bias = [np.random.normal(0,0.1, layers[i]) for i in range(len(layers))]
weights = init_weights(X[0], layers)

err_vector = []

print("start")
for epoch in range(epochs):
    count = 0
    err = 0

    for my_x in X:
        #feed_forward
        a = feed_forward(my_x, layers, weights, bias)


        #Net error
        err += net_error(tar[count], a[len(a) -1])

        #back propagation
        weights, bias = back_propagation(my_x, tar[count], weights, bias, a)

        count +=1

    err_vector.append(err / X.shape[0])

#graph error
graph_error(err_vector)

#testing patterns
testing_patterns(X, tar, layers, weights, bias)
