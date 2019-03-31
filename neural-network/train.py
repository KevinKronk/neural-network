import numpy as np
from scipy import optimize as opt

from back_propagation import back_propagation
from cost_function import cost_function
from cost_function import unroll_params
from feed_forward import feed_forward


def train(init_params, x, y, y_map, hyper_p=0, iters=100):
    """ Trains a Neural Network, compares to the labelled outputs and reports an accuracy. """

    result = opt.minimize(cost_function, init_params, args=(x, y_map, hyper_p),
                          jac=back_propagation, method='TNC', options={'maxiter': iters})

    theta1, theta2 = unroll_params(result.x)
    prediction = np.argmax(feed_forward([theta1, theta2], x)[1][1], axis=0)
    accuracy = np.mean(prediction == y.T) * 100

    return accuracy
