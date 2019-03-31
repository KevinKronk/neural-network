from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy import optimize as opt
from load_data import load_data
from load_weights import load_weights
from cost_function import cost_function
from map_outputs import map_outputs


# Load Data and Weights

filename = 'ex4data1.mat'
x, y = load_data(filename)

weight_file = 'ex4weights.mat'
params = load_weights(weight_file)


# Visualize the Images

nrows = 4
ncols = 4


def create_image(x_row):
    return np.reshape(x_row, (20, 20)).T


indices = np.random.randint(x.shape[0], size=nrows * ncols)
images = x[indices, :]

fig, ax = plt.subplots(nrows, ncols, figsize=(10, 8))

c = 0
for j in range(nrows):
    for i in range(ncols):
        ax[j][i].imshow(create_image(images[c, :]), cmap='gray')
        ax[j][i].axis('off')
        c += 1
# plt.show()


# Set values and Load Weights

input_layer = 400
hidden_layer = 25
output_layer = 10


# Map Outputs

k = 10
y_map = map_outputs(y, k)


cost = cost_function(params, x, y_map, hyper_p=0)

print(cost)


def sig_gradient(z):
    sig_grad = expit(z) * (1 - expit(z))
    return sig_grad


def rand_weight(lin, lout):
    eps = np.sqrt(6) / np.sqrt(lin + lout)
    w = np.random.rand(lout, 1 + lin) * 2 * eps - eps
    return w


init_theta1 = rand_weight(input_layer, hidden_layer)
init_theta2 = rand_weight(hidden_layer, output_layer)
init_params = np.concatenate([init_theta1.ravel(), init_theta2.ravel()], axis=0)


def back_propagation(params, x, y_map, hyper_p=0):
    m = len(x)

    theta1, theta2 = unroll_params(params)
    D1 = np.zeros_like(theta1)
    D2 = np.zeros_like(theta2)

    history = feed_forward([theta1, theta2], x)
    a3 = history[1][1]
    z3 = history[1][0]
    a2 = history[0][1]
    z2 = history[0][0]
    a1 = np.insert(x, 0, 1, axis=1)

    delta3 = a3 - y_map
    delta2 = (theta2.T @ delta3)[1:] * sig_gradient(z2)

    D1 = (D1 + delta2 @ a1) / m
    D2[:, 1:] = (D2[:, :1] + delta3 @ a2.T) / m

    D1[:, 1:] = D1[:, 1:] + (hyper_p / m) * theta1[:, 1:]
    D2[:, 1:] = D2[:, 1:] + (hyper_p / m) * theta2[:, 1:]

    return np.concatenate([D1.ravel(), D2.ravel()])


# result = back_propagation(init_thetas, x, y_map)
# print(result)


def train(init_params, x, y_map, hyper_p=0, iters=100):
    result = opt.minimize(cost_function, init_params, args=(x, y_map, hyper_p),
                          jac=back_propagation, method='TNC', options={'maxiter': iters})
    return result.x


hyper_p = 10

params = train(init_params, x, y_map, hyper_p, 150)
theta1, theta2 = unroll_params(params)
prediction = np.argmax(feed_forward([theta1, theta2], x)[1][1], axis=0)

accuracy = np.mean(prediction == y.T) * 100
print(f"With a hyperparameter of {hyper_p}\nThe accuracy is: {accuracy}%")
