import numpy as np
import matplotlib.pyplot as plt


def one_hot_encoding(y, outputs=10):
    return np.eye(outputs)[y]


def show_image(norm_pixels, label=''):
    plt.title('Label is {label}'.format(label=np.argmax(label)))
    plt.imshow(np.array(norm_pixels * 255, dtype='uint8').reshape((28, 28)), cmap='gray')
    plt.show()
