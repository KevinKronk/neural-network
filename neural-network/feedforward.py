from load_data import load_data
from scipy.io import loadmat
from prediction import predict
import numpy as np
import matplotlib.pyplot as plt

# Load Data

filename = "ex3data1.mat"
x, y = load_data(filename)
y = y.ravel()

weights = "ex3weights.mat"
data = loadmat(weights)
theta1 = data.get('Theta1')
theta2 = data.get('Theta2')
theta2 = np.roll(theta2, 1, axis=0)

prediction = predict(theta1, theta2, x)
accuracy = np.mean(prediction == y)
print(f"{round(accuracy * 100, 2)}%")


def row_to_square(X_row):
    return np.reshape(X_row, (20, 20)).T


size = x.shape[0]
indices = np.random.randint(size, size=3)


for index in indices:
    predicted_value = predict(theta1, theta2, x[index, :])
    actual_value = y[index]
    plt.imshow(row_to_square(x[index, :]), cmap='gray')
    plt.title(f"Predicted: {int(predicted_value)}", size=15)
    plt.xlabel(f"Actual: {actual_value}", size=15)
    plt.show()

