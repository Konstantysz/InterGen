from numba import jit
import numpy as np

@jit(nopython=True, parallel=True)
def generateSphericalObject(X, Y, x0 = 0, y0 = 0, f = 1, h = 0):
    return f*(np.power(X - x0,2) + np.power(Y - y0,2)) + h