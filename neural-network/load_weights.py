from scipy.io import loadmat
import numpy as np


def load_weights(filename):
    weights = loadmat(filename)

    theta1 = weights['Theta1']
    theta2 = weights['Theta2']
    theta2 = np.roll(theta2, 1, axis=0)

    params = np.concatenate([theta1.ravel(), theta2.ravel()])
    return params
