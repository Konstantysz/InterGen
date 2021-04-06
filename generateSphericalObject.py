import matplotlib.pyplot as plt
import numpy as np

def generateSphericalObject(imSize):
    x = np.linspace(-1, 1, imSize)
    y = np.linspace(-1, 1, imSize)
    X, Y = np.meshgrid(x, y)

    res = 2*np.power(X,2) + 2*np.power(Y,2)

    # Z2 = np.full((imSize, imSize), 1) - (np.power(X,2) + np.power(Y,2))
    # Z2 = (Z2 + 1) / 2
    # res = np.sqrt(Z2)
    # plt.imshow(res)
    # plt.show()

    return res