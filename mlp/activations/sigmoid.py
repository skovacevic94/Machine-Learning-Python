import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    sigmoid_value = sigmoid(z)
    return np.multiply(sigmoid_value, (1-sigmoid_value))