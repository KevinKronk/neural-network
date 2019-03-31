import numpy as np


def map_outputs(y, k):
    """ Maps integer output labels to binary. """

    m = y.size
    y_map = np.zeros((k, m))
    for i in range(m):
        y_map[y[i], i] = 1

    return y_map
