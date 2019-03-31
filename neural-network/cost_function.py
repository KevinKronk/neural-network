import numpy as np
from feed_forward import feed_forward


input_layer = 400
hidden_layer = 25
output_layer = 10


def unroll_params(params):
    theta1 = np.reshape(params[:hidden_layer * (input_layer + 1)],
                        (hidden_layer, (input_layer + 1)))
    theta2 = np.reshape(params[(hidden_layer * (input_layer + 1)):],
                        (output_layer, (hidden_layer + 1)))
    return theta1, theta2


def cost_function(params, x, y_map, hyper_p=0):
    size = len(x)
    thetas = np.array(unroll_params(params))
    h = feed_forward(thetas, x)[1][1]

    first = -y_map * np.log(h)
    second = (1 - y_map) * np.log(1 - h)
    cost = np.sum((first - second) / size)
    reg = (hyper_p / (2 * size)) * (np.sum(thetas[0] ** 2) + np.sum(thetas[1] ** 2))

    return cost + reg
