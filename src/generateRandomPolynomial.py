from numba import jit
import numpy as np

@jit(nopython=True, parallel=True)
def generateRandomPolynomial(X, Y, n):
    '''
    Function that generetes 2D discrete polynomial function distribution of specific order.
    Boosted with Numba: works in C and with parallel computing.

    Parameters
    ----------
    X : numpy.ndarray
        meshgrided values in X axis
    Y : numpy.ndarray
        meshgrided values in Y axis
    n : int
        order of polynomial
        
    Returns:
    ----------
    res : numpy.ndarray
        matrix of 2D polynomial function distribution
    '''
    a = (2 * np.random.random_sample((3*n,)) - 1) * np.eye(n*3)

    XY = []

    if n > 0:
        for i in range(n, 0, -1):
            XY.append(X**i)
            XY.append(Y**i)
            XY.append(X**(i-1)*Y**(i-1))
    elif n == 0:
        XY.append(X)
        XY.append(Y)
        XY.append(X**(0)*Y**(0))

    res = np.zeros(X.shape)
    for i in range(3*n):
        res = res + a[i, i] * XY[i]

    return res