import numpy as np

from activations import relu_activation


class NeuralNetwork():

    def __init__(self, layers, batch_generator, activation=relu_activation):

        # batch_generator
        self.batch_generator = batch_generator

        # activation function
        self.activation = activation

        # architecture
        self.layers = layers

        # number of layers
        self.n_layers = len(layers)

        # init weights
        self.weights = []
        for i in range(self.n_layers - 1):
            self.weights.append(0.005 * np.random.rand(layers[i], layers[i + 1]))

        # init biases
        self.biases = [0.005 * np.random.rand(b) for b in layers[1:]]

    def _forward_pass(self, input):

        # inputs to the layers
        inputs = []

        # activations of layers. (input data is the activation of the input layer)
        activations = [input]

        # forward calculations
        for i in range(self.n_layers - 1):
            inputs.append(activations[i].dot(self.weights[i]) + self.biases[i])
            activations.append(self.activation['transfer']((inputs[-1])))

        return {'inputs': inputs, 'activations': activations}

    def _backward_pass(self, forward, y):

        # delta placeholders
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]

        # error output layer
        error = forward['activations'][-1] - y

        # deltas output layer
        delta_w[-1] += forward['activations'][-2].transpose().dot(error * self.activation['diff'](forward['inputs'][-1]))
        delta_b[-1] += np.sum(error, axis=0)

        # error hidden layer
        for l in range(self.n_layers - 3, -1, -1):
            error = error.dot(self.weights[l + 1].transpose())
            # deltas hidden layer
            delta_w[l] += forward['activations'][l].transpose().dot(error * self.activation['diff'](forward['inputs'][l]))
            delta_b[l] += np.sum(error, axis=0)

        return delta_w, delta_b

    def _evaluation(self, x, y):

        # placeholders
        predictions = []
        true_labels = []

        # inference
        for i in range(x.shape[0]):
            predictions.append(self.inference(np.atleast_2d(x[i])))
            true_labels.append(y[i])

        return predictions, true_labels

    def train(self, learning_rate=0.001):

        # loop over all training example
        for e in range(int(len(self.batch_generator.training_data) / self.batch_generator.batch_size)):
            # batch data
            x_batch, y_batch = self.batch_generator.batch()

            # deltas
            delta_w, delta_b = self._backward_pass(self._forward_pass(x_batch), y_batch)

            # update weights and biases
            for l in range(self.n_layers - 1):
                self.weights[l] -= learning_rate * delta_w[l]
                self.biases[l] -= learning_rate * delta_b[l]

    def validation(self):
        return self._evaluation(*self.batch_generator.validation_data)

    def test(self):
        return self._evaluation(*self.batch_generator.test_data)

    def inference(self, input):
        # forward pass with argmax on output layer
        return np.argmax(self._forward_pass(input)['activations'][-1])
