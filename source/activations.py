import numpy as np

lrelu_activation = {'transfer': lambda x: np.maximum(x, 0.1 * x), 'diff': lambda x: (x < 0) * 0.1 + (x >= 0)}

relu_activation = {'transfer': lambda x: np.maximum(x, 0), 'diff': lambda x: (x > 0)}

tanh_activation = {'transfer': lambda x: np.tanh(x), 'diff': lambda x: 1.0 - np.tanh(x)**2}

sigmoid_activation = {'transfer': lambda x: 1.0 / (1.0 + np.exp(-x)),
                      'diff': lambda x: 1.0 / (1.0 + np.exp(-x)) * (1 - 1.0 / (1.0 + np.exp(-x)))}
