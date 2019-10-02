import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt

class MLP:
    """Class defining a neural network using back propagation"""

    def __init__(self, layers):
        """layers is initialized as a list(input_layer, ...hidden_layers..., output_layers)"""
        self.n_layers = len(layers) - 1
        self.layers = layers
        self.initialize_weights()
        self.actual_output = 0

    def initialize_weights(self):
        """generate weights and biases for hidden layers
        in a standard gaussian distribution  mean 0 deviation 1"""
        self.biases = [np.random.randn(1, y) for y in self.layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])]

    def sigmoid(self, x):
        """The sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def derivative(self, sig_x):
        """return the derivative of sig_x"""
        return sig_x*(1 - sig_x)

    def net_error(self, tar, out):
        """compute the network error"""
        err = 0.5*np.power(tar - out, 2)

        return err

    def graph_error(self, err_vector):
        """display graphical error"""
        plt.figure(0)
        plt.plot(err_vector)
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.title("Back-Propagation algortihm")
        plt.show()

    def testing_patterns(self, input, target):
        """return the response for the patterns"""
        print('====== MLP result ======')
        print('Pat:    t:    out:')
        count = 0
        for i in input:
            output = self.feed_forward(i)[-1][0]
            if not hasattr(target[count], '__len__'):
                print('{}. {} ---- {} ----> {:.3f}'.format(count, i, target[count], float(output)))
            else:
                count_2 = 0
                for o in output:
                    print('{}. {} ---- {} ----> {:.3f}'.format(count, i, target[count][count_2], float(o)))
                    count_2 += 1
            count += 1

    def feed_forward(self, input):
        """the feed forward function"""
        self.layers_output = []

        for index in range(self.n_layers):
            if index == 0:
                z = np.dot(input, self.weights[index].T) + self.biases[index]
                self.layers_output.append(self.sigmoid(z))
            else:
                z = np.dot(self.layers_output[index -1], self.weights[index].T) + self.biases[index]
                self.layers_output.append(self.sigmoid(z))

        return self.layers_output

    def back_propagation(self, input, target, trainingRate = 0.2):
        """the back_propagation function"""
        l_errors = []

        #feed forward
        self.feed_forward(input)

        # compute l_errors
        for index in reversed(range(self.n_layers)):
            if index == self.n_layers - 1:
                sig_prim = self.derivative(self.layers_output[index])
                output_delta = -(target - self.layers_output[index])*sig_prim
                l_errors.append(output_delta)
            else:
                hidden_delta = np.dot(l_errors[-1], self.weights[index+1])
                l_errors.append(hidden_delta*self.derivative(self.layers_output[index]))

        l_errors = l_errors[::-1]
        self.layers_output.insert(0, input)

        # new biases and weights
        for index in range(self.n_layers ):
            multiply = np.multiply(l_errors[index].T, self.layers_output[index])
            self.weights[index] = self.weights[index] - trainingRate*multiply
            self.biases[index] = self.biases[index] - trainingRate*l_errors[index]

        return (self.weights, self.biases)

    def main(self, epochs, trainingRate, input, target):
        err_vector = []

        for epoch in range(epochs):
            count = 0
            err = 0

            for x in input:

                #back propagation
                self.back_propagation(x, target[count], trainingRate)

                #net error
                err+= self.net_error(target[count], self.layers_output[-1][0])

                count +=1
            err_vector.append(err / input.shape[0])

        #graph error
        self.graph_error(err_vector)

        #testings patterns
        self.testing_patterns(input, target)
