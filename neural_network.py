import numpy as np

from utils import relu, relu_diff


class NeuralNetwork():
    def __init__(self, layers):
        # layers
        self.layers = layers

        # number of layers
        self.n_layers = len(layers)

        # init weights
        self.weights = []
        for i in range(self.n_layers - 1):
            self.weights.append(0.005 * np.random.rand(layers[i], layers[i + 1]))

        # init biases
        self.biases = [0.005 * np.random.rand(b) for b in layers[1:]]

    def _forward_pass(self, input, verbose=False):
        inputs = []
        activations = [input]
        for i in range(self.n_layers - 1):
            inputs.append(activations[i].dot(self.weights[i]) + self.biases[i])
            activations.append(relu(inputs[-1]))
        return {'inputs': inputs, 'activations': activations}

    def _backward_pass(self, forward, y):

        # delta placeholders
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]

        # error output layer
        error = forward['activations'][-1] - y
        # deltas output layer
        delta_w[-1] += forward['activations'][-2].transpose().dot(error * relu_diff(forward['inputs'][-1]))
        delta_b[-1] += np.sum(error, axis=0)

        # hidden layers
        for l in range(self.n_layers - 3, -1, -1):
            # error hidden layer
            error = error.dot(self.weights[l + 1].transpose())
            # deltas hidden layer
            delta_w[l] += forward['activations'][l].transpose().dot(error * relu_diff(forward['inputs'][l]))
            delta_b[l] += np.sum(error, axis=0)

        return delta_w, delta_b

    def train(self, batch_generator, learning_rate=0.02, epochs=60000):

        for e in range(epochs):
            # batch
            x_batch, y_batch = batch_generator.batch()

            # deltas
            delta_w, delta_b = self._backward_pass(self._forward_pass(x_batch), y_batch)

            # update weights and biases
            for l in range(self.n_layers - 1):
                self.weights[l] -= learning_rate * delta_w[l]
                self.biases[l] -= learning_rate * delta_b[l]

    def inference(self, input):
        return np.argmax(self._forward_pass(input)['activations'][-1])
