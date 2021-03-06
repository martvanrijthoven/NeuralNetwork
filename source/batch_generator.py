import numpy as np
import _pickle as cPickle
import gzip
import random

from utils import one_hot_encoding


class BatchGenerator():
    def __init__(self, batch_size):

        # load mnist data set
        self.training_data, self.validation_data, self.test_data = \
            cPickle.load(gzip.open('../data/mnist.pkl.gz', 'rb'), encoding='iso-8859-1')

        # set up training data with one hot encoding
        self.training_data = list(zip(self.training_data[0], one_hot_encoding(self.training_data[1])))

        # number of examples in batch
        self.batch_size = batch_size

        # index for batch training data
        self._batch_index = 0

        # reset training data
        self._reset()

    def _shuffle(self):

        # shuffle training data
        random.shuffle(self.training_data)

    def _reset(self):

        # shuffle training data
        self._shuffle()

        # reset batch training index
        self._batch_index = 0

    def batch(self):

        # batch data
        x_batch, y_batch = zip(*self.training_data[self._batch_index:self._batch_index + self.batch_size])

        # increase batch index
        self._batch_index += self.batch_size

        # reset if batch index exceeds number of training examples
        if self._batch_index >= len(self.training_data):
            self._reset()

        return np.array(x_batch), np.array(y_batch)
