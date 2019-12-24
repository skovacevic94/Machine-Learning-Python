import numpy as np

def mean_squared_error(y, y_true):
    return (np.square(y - y_true)).mean(axis=0)