import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from activations import relu, relu_diff


class NeuralNetwork():

    def __init__(self, layers):

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

    def _forward_pass(self, input, verbose=False):

        # inputs to the layers
        inputs = []

        # activations of layers. (input data is the activation of the input layer)
        activations = [input]

        # forward calculations
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

        # error hidden layer
        for l in range(self.n_layers - 3, -1, -1):
            error = error.dot(self.weights[l + 1].transpose())
            # deltas hidden layer
            delta_w[l] += forward['activations'][l].transpose().dot(error * relu_diff(forward['inputs'][l]))
            delta_b[l] += np.sum(error, axis=0)

        return delta_w, delta_b

    def train(self, batch_generator, learning_rate=0.02):

        # epoch loop
        for e in range(int(len(batch_generator.training_data) / batch_generator.batch_size)):
            # batch data
            x_batch, y_batch = batch_generator.batch()

            # deltas
            delta_w, delta_b = self._backward_pass(self._forward_pass(x_batch), y_batch)

            # update weights and biases
            for l in range(self.n_layers - 1):
                self.weights[l] -= learning_rate * delta_w[l]
                self.biases[l] -= learning_rate * delta_b[l]

    def validation(self, batch_generator):

        # validation data
        x_val, y_val = batch_generator.validation_data

        # placeholders
        predictions = []
        true_labels = []

        # inferece
        for i in range(x_val.shape[0]):
            predictions.append(self.inference(np.atleast_2d(x_val[i])))
            true_labels.append(np.argmax(y_val[i]))

        return accuracy_score(true_labels, predictions)

    def test(self, batch_generator):

        # test data
        x_test, y_test = batch_generator.test_data

        # placeholders
        predictions = []
        true_labels = []

        for i in range(x_test.shape[0]):
            predictions.append(self.inference(np.atleast_2d(x_test[i])))
            true_labels.append(np.argmax(y_test[i]))

        return confusion_matrix(true_labels, predictions), classification_report(true_labels, predictions)

    def inference(self, input):

        # forward pass with argmax on output layer
        return np.argmax(self._forward_pass(input)['activations'][-1])
