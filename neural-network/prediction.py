import numpy as np
from scipy.special import expit


def predict(theta1, theta2, x):
    if x.ndim == 1:
        x = x[None]
    a1 = np.insert(x, 0, 1, axis=1)
    z2 = theta1 @ a1.T
    a2 = expit(z2)
    a2 = np.insert(a2, 0, 1, axis=0)
    z3 = theta2 @ a2
    a3 = expit(z3)
    return np.argmax(a3, axis=0)
