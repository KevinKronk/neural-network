from scipy.io import loadmat


def load_data(filename):
    """ Loads a .mat file into an ndarray. """

    data = loadmat(filename)

    x = data['X']
    y = data['y']

    y[y == 10] = 0

    return x, y
