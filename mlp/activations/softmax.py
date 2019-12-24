import numpy as np

def softmax(Z):
    e = np.exp(Z)
    sum_e = np.sum(e, axis=1, keepdims=True)
    return e / sum_e


