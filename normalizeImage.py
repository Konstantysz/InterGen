from numba import jit
import numpy as np
import matplotlib.pyplot as plt

# @jit(nopython=True, parallel=True)
def normalizeImage(I, normFactor = 255.0):
    normI = (I * (normFactor/np.max(np.abs(I))) + normFactor) / 2
    return normI