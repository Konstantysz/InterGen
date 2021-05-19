from numba import jit
import numpy as np

@jit(nopython=True, parallel=True)
def gauss_n(X, Y):
    amp = 1.0
    sigma = 3.0
    mu = 0.0

    exponent = ((X - mu)**2 + (Y - mu)**2) / 2*sigma
    val = (amp*np.exp(-exponent))
    
    return val