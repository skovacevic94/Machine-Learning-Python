import numpy as np
import numpy.linalg as linalg

def gaussian_pdf(x, mean, variance):
    return (1/np.sqrt(2*np.pi*variance))*np.exp(-np.square(x-mean)/(2*variance))

def multivariate_gaussian_pdf(x, mean, sigma):
    num_features = x.shape[1]
    if num_features == mean.shape[0] and (num_features, num_features) == sigma.shape:
        det = linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0 / ( np.power((2*np.pi),float(num_features)/2) * np.power(det,1.0/2) )
        x_mu = np.matrix(x - mean)
        inv = linalg.inv(sigma)
        scores = np.sum(np.multiply(np.matmul(x_mu, inv), x_mu), axis=1)
        result = np.exp(-0.5*scores) # #math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")
