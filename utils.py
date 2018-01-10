import numpy as np
import matplotlib.pyplot as plt


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
