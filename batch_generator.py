import numpy as np
import _pickle as cPickle
import gzip
import random

from utils import one_hot_encoding


class BatchGenerator():
    def __init__(self, batch_size):
        mnist = gzip.open('./data/mnist.pkl.gz', 'rb')
        self.training_data, self.validation_data, self. test_data = \
            cPickle.load(mnist, encoding='iso-8859-1')

        self.training_data = list(zip(self.training_data[0], one_hot_encoding(self.training_data[1])))
        self.validation_data = (self.validation_data[0], one_hot_encoding(self.validation_data[1]))
        self.test_data = (self.test_data[0], one_hot_encoding(self.test_data[1]))

        self.batch_size = batch_size
        self.batch_index = 0
        self.reset()

    def _shuffle(self):
        self.batch_index = 0
        random.shuffle(self.training_data)

    def batch(self):
        x_batch, y_batch = zip(*self.training_data[self.batch_index:self.batch_index + self.batch_size])

        self.batch_index += self.batch_size

        if self.batch_index >= len(self.training_data):
            self.reset()

        return np.array(x_batch), np.array(y_batch)

    def reset(self):
        self._shuffle()
        self.batch_index = 0
