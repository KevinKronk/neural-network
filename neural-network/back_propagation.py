import numpy as np

from additional_funtions import sig_gradient
from cost_function import unroll_params
from feed_forward import feed_forward


def back_propagation(params, x, y_map, hyper_p=0):
    m = len(x)

    theta1, theta2 = unroll_params(params)
    d1 = np.zeros_like(theta1)
    d2 = np.zeros_like(theta2)

    history = feed_forward([theta1, theta2], x)
    a3 = history[1][1]
    a2 = history[0][1]
    z2 = history[0][0]
    a1 = np.insert(x, 0, 1, axis=1)

    delta3 = a3 - y_map
    delta2 = (theta2.T @ delta3)[1:] * sig_gradient(z2)

    d1 = (d1 + delta2 @ a1) / m
    d2[:, 1:] = (d2[:, :1] + delta3 @ a2.T) / m

    d1[:, 1:] = d1[:, 1:] + (hyper_p / m) * theta1[:, 1:]
    d2[:, 1:] = d2[:, 1:] + (hyper_p / m) * theta2[:, 1:]

    return np.concatenate([d1.ravel(), d2.ravel()])
