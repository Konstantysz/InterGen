from numba import jit
import numpy as np

@jit(nopython=True, parallel=True)
def gauss_n(X, Y, mu_x = 0.0, mu_y = 0.0, amp = 1.0, sigma = 3.0):
    '''
    Function that generates 2D discrete gaussian distribution.
    Boosted with Numba: works in C and with parallel computing.

    Parameters
    ----------
    X : numpy.ndarray
        meshgrided values in X axis
    Y : numpy.ndarray
        meshgrided values in Y axis
    mu_x : float
        Displacement in X axis
    mu_y : float
        Displacement in Y axis
    amp : float
        Amplitude of gaussian distribution
    sigma : float
        Std dev of gaussian distribution
        
    Returns:
    ----------
    val : numpy.ndarray
        matrix of 2D gaussian distribution
    '''
    exponent = ((X - mu_x)**2 + (Y - mu_y)**2) / 2*sigma
    val = (amp*np.exp(-exponent))
    
    return val