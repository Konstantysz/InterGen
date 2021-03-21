import numpy as np

def generateRandomPolynomial(imSize, n):
    x = np.linspace(-1, 1, imSize)
    y = np.linspace(-1, 1, imSize)
    X, Y = np.meshgrid(x, y)

    a = (10 * np.random.random_sample((3*n,)) - 5) * np.eye(n*3)

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

    XY = np.array(XY)
    res = np.zeros((imSize, imSize))
    for i in range(9):
        res = res + a[i, i] * XY[i]

    return res