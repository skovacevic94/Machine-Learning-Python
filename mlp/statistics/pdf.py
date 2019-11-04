import numpy as np

def gaussian_pdf(x, mean, variance):
    return (1/np.sqrt(2*np.pi*variance))*np.exp(-np.square(x-mean)/(2*variance))
