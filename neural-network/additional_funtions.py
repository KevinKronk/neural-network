import numpy as np
from scipy.special import expit


def rand_weight(lin, lout):
    eps = np.sqrt(6) / np.sqrt(lin + lout)
    w = np.random.rand(lout, 1 + lin) * 2 * eps - eps
    return w


def sig_gradient(z):
    sig_grad = expit(z) * (1 - expit(z))
    return sig_grad
