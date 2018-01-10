import numpy as np


#  relu functions
def relu(x):
    return np.maximum(x, 0)


def relu_diff(x):
    return (x > 0)


# sigmoid functions
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_diff(x):
    return sigmoid(x) * (1 - sigmoid(x))


# tanh functions
def tanh(x):
    return np.tanh(x)


def tanh_diff(x):
    return 1.0 - np.tanh(x)**2
