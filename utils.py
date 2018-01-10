import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(x, 0)


def relu_diff(x):
    return (x > 0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_diff(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_diff(x):
    return 1.0 - np.tanh(x)**2


def one_hot_encoding(y, outputs=10):
    labels = np.zeros((len(y), outputs))
    for l in range(len(y)):
        labels[l][y[l]] = 1
    return labels


def show_image(norm_pixels, label=None):
    if label is not None:
        plt.title('Label is {label}'.format(label=np.argmax(label)))
    img = np.array(norm_pixels * 255, dtype='uint8').reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.show()
