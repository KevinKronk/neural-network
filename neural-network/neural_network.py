import matplotlib.pyplot as plt
import numpy as np

from additional_funtions import rand_weight
from load_data import load_data
from load_weights import load_weights
from map_outputs import map_outputs
from train import train


# Load Data and Weights

filename = 'ex4data1.mat'
x, y = load_data(filename)

weight_file = 'ex4weights.mat'
params = load_weights(weight_file)


# Set values

input_layer = 400
hidden_layer = 25
output_layer = 10


# Map Outputs

k = 10
y_map = map_outputs(y, k)


# Initialize Randomized Parameters

init_theta1 = rand_weight(input_layer, hidden_layer)
init_theta2 = rand_weight(hidden_layer, output_layer)
init_params = np.concatenate([init_theta1.ravel(), init_theta2.ravel()], axis=0)
hyper_p = 10
iters = 150


# Train and Predict

accuracy = train(init_params, x, y, y_map, hyper_p, iters)

print(f"With a hyperparameter of {hyper_p}\nThe accuracy is: {accuracy}%")


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
plt.show()
