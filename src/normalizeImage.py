from numba import jit
import numpy as np
import matplotlib.pyplot as plt

@jit(nopython=True, parallel=True)
def normalizeImage(I, normFactor = 255.0):
    '''
    Function that normalize matrix to a range of values defined by user.
    Boosted with Numba: works in C and with parallel computing.

    Parameters
    ----------
    I : numpy.ndarray
        matrix to be normalized
    normFactor : float
        max value
        
    Returns:
    ----------
    normI : numpy.ndarray
        matrix of normalized values
    '''
    normI = (I * (normFactor / np.max(np.abs(I))) + normFactor) / 2
    return normI