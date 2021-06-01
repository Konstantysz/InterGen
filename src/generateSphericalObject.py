from numba import jit
import numpy as np

@jit(nopython=True, parallel=True)
def generateSphericalObject(X, Y, x0 = 0, y0 = 0, f = 1, h = 0):
    '''
    Function that generetes 2D discrete spherical function distribution.
    Boosted with Numba: works in C and with parallel computing.

    Parameters
    ----------
    X : numpy.ndarray
        meshgrided values in X axis
    Y : numpy.ndarray
        meshgrided values in Y axis
    x0 : int
        center of sphere in X axis
    y0 : int
        center of sphere in Y axis
    h : int
        height of sphere
        
    Returns:
    ----------
    res : numpy.ndarray
        matrix of 2D spherical function distribution
    '''
    return f*(np.power(X - x0,2) + np.power(Y - y0,2)) + h