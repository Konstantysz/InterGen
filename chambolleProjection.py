import matplotlib.pyplot as plt
from numba import jit
import numpy as np
from skimage.restoration import denoise_tv_chambolle
import time

from normalizeImage import normalizeImage

# @jit(nopython=True, parallel=True)
def divergence(f):
    """
    Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
    :param f: List of ndarrays, where every item of the list is one dimension of the vector field
    :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
    """
    num_dims = len(f)
    grad = [np.gradient(f[i], axis=i) for i in range(num_dims)]
    return grad[0] + grad[1]

def chambolleProjection(f, f_ref, bg_ref, iterations = 2000, mi = 100, tau = 0.25):
    xi = np.array([np.zeros(f.shape), np.zeros(f.shape)])
    x1 = np.zeros(f.shape)
    x2 = np.zeros(f.shape)
    x_best = np.zeros(f.shape)

    err_n = 0.0
    err = []
    pp = []
    pr = 1.0

    err_min = 1.0
    err_min_it = 0    

    for i in range(iterations):
        gdv = np.array(np.gradient(divergence(xi) - f/mi))
        d = np.sqrt(np.power(gdv[0], 2) + np.power(gdv[1], 2))
        d = np.tile( d, [2, 1, 1] )
        xi = np.divide(xi + tau * gdv, 1 + tau * d)

        x2 = mi * divergence(xi)
        diff_err = normalizeImage(x2) - f_ref
        err_i = np.sqrt(np.mean(np.power(diff_err, 2))) / np.sqrt(np.mean(np.power(f_ref, 2)))
        # err_i = np.linalg.norm(x2 - f_ref) / np.linalg.norm(f_ref)
        err.append(err_i)
        # err.append(np.linalg.norm(x2 - x1) / np.linalg.norm(f))
        # g_err = np.abs((err_n - err[i]) / 2)
        # err_n = err[i]
        # pp.append(g_err / err[0])
        # pr = pp[i]

        if err[i] < err_min:
            err_min = err[i]
            err_min_it = i
            x_best = x2
            print("Minimal error: {}, iteration #{}".format(err_min, err_min_it))
            if err_min_it == iterations - 1:
                iterations = iterations + 100

        x1 = x2

    x_best = normalizeImage(x_best)

    plt.subplot(2, 2, 1)
    plt.imshow(f)
    plt.subplot(2, 2, 2)
    plt.imshow(x_best)
    plt.subplot(2, 2, 3)
    plt.imshow(f_ref)
    plt.subplot(2, 2, 4)
    plt.imshow(bg_ref)
    plt.show()

    plt.plot(err)
    plt.show()

    return x_best