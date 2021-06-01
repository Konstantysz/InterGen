from numba import jit
import numpy as np

@jit(nopython=True, parallel=True)
def gauss_n(X, Y):
    '''
    Function that generates 2D discrete gaussian distribution.
    Boosted with Numba: works in C and with parallel computing.

    Parameters
    ----------
    X : numpy.ndarray
        meshgrided values in X axis
    Y : numpy.ndarray
        meshgrided values in Y axis
        
    Returns:
    ----------
    val : numpy.ndarray
        matrix of 2D gaussian distribution
    '''
    amp = 1.0
    sigma = 3.0
    mu = 0.0

    exponent = ((X - mu)**2 + (Y - mu)**2) / 2*sigma
    val = (amp*np.exp(-exponent))
    
    return val