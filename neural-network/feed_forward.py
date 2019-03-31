import numpy as np
from scipy.special import expit


def feed_forward(thetas, x):
    """ Performs the feed-forward algorithm and saves each step in an array. """

    a = np.copy(x.T)
    history = []

    for theta in thetas:
        a = np.insert(a, 0, 1, axis=0)
        z = theta @ a
        a = expit(z)
        history.append((z, a))

    return history
