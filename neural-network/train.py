from cost_function import cost_function
from back_propagation import back_propagation
from scipy import optimize as opt


def train(init_params, x, y_map, hyper_p=0, iters=100):
    result = opt.minimize(cost_function, init_params, args=(x, y_map, hyper_p),
                          jac=back_propagation, method='TNC', options={'maxiter': iters})
    return result.x
