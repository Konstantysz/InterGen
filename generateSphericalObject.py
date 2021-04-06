import matplotlib.pyplot as plt
import numpy as np

def generateSphericalObject(imSize, x0 = 0, y0 = 0, f = 1, h = 0):
    x = np.linspace(-1, 1, imSize)
    y = np.linspace(-1, 1, imSize)
    X, Y = np.meshgrid(x, y)

    return f*(np.power(X - x0,2) + np.power(Y - y0,2)) + h