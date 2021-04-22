import time
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import denoise_tv_chambolle

from normalizeImage import normalizeImage

def divergence(f):
    """
    Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
    :param f: List of ndarrays, where every item of the list is one dimension of the vector field
    :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
    """
    num_dims = len(f)
    grad = [np.gradient(f[i], axis=i) for i in range(num_dims)]
    return grad[0] + grad[1]

def chambolleProjection(f, f_ref, bg_ref, iterations = 1000, mi = 100, tau = 0.25, tol = 1e-6):

    print(np.min(f))
    print(np.max(f))
    print(np.min(f_ref))
    print(np.max(f_ref))

    if np.max(f_ref) != 1 and np.min(f_ref) != 0:
        f_ref = normalizeImage(f_ref, normFactor = 1)
    if np.max(f) != 1 and np.min(f) != 0:
        f = normalizeImage(f, normFactor = 1)
    
    print(np.min(f))
    print(np.max(f))
    print(np.min(f_ref))
    print(np.max(f_ref))
    
    # plt.imshow(f)
    # plt.show()
    # plt.imshow(f_ref)
    # plt.show()

    n = 1
    xi = np.array([np.zeros(f.shape), np.zeros(f.shape)])
    x1 = np.zeros(f.shape)
    x2 = np.zeros(f.shape)
    x_best = np.zeros(f.shape)

    rms_min_A = []
    rms_min = 1.0
    it_min = 0    

    start_time = time.time()
    while n - it_min < 100:
        gdv = np.array(np.gradient(divergence(xi) - f/mi))
        d = np.sqrt(np.power(gdv[0], 2) + np.power(gdv[1], 2))
        d = np.tile( d, [2, 1, 1] )
        xi = np.divide(xi + tau * gdv, 1 + tau * d)

        x2 = mi * divergence(xi)
        
        diff = x2 - f_ref

        rms_n = np.sqrt(np.var(diff))
        
        if len(rms_min_A) < 100:
            rms_min_A.append(rms_min)
        else:
            rms_min_A.pop(0)
            rms_min_A.append(rms_min)

        if rms_n < rms_min:
            rms_diff = rms_min_A[0] - rms_min_A[-1]
            rms_local_diff = rms_min - rms_n

            if ((rms_diff < 10 * tol) and (rms_local_diff < tol)):
                rms_min = rms_n
                it_min = n
                break
            rms_min = rms_n
            it_min = n

        x1 = x2
        n = n + 1

    x_best = x2

    print("RMS = {}, Itarations: {}".format(rms_min, it_min))
    print("--- {} seconds ---".format(time.time() - start_time))

    plt.subplot(2, 2, 1)
    plt.title("Input For VID")
    plt.imshow(f)
    plt.subplot(2, 2, 2)
    plt.title("Output of Chambolle Projection with smallest error")
    plt.imshow(x_best)
    plt.subplot(2, 2, 3)
    plt.title("Texture function")
    plt.imshow(f_ref)
    plt.subplot(2, 2, 3)
    plt.title("Input - Output")
    plt.imshow(f - x_best)
    plt.subplot(2, 2, 4)
    plt.title("Background function")
    plt.imshow(bg_ref)
    plt.show()

    return x_best