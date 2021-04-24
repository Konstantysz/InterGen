from numba import jit
import time
import matplotlib.pyplot as plt
import numpy as np

def gradient2D(mat):
    x1 = mat[1:len(mat), :]
    x2 = np.array([mat[-1, :]])
    x = np.concatenate((x1, x2), axis = 0)

    fx = x - mat
    y1 = mat[:, 1:len(mat)]
    y2 = np.array([mat[:, -1]]).transpose()
    y = np.concatenate((y1, y2), axis = 1)

    fy = y - mat
    return [fx, fy]

def divergence2D(mat):
    Px = mat[0]
    Py = mat[1]

    fx = Px - np.concatenate((np.array([Px[0, :]]), Px[0:len(Px)-1, :]), axis=0)
    fx[0, :] = Px[0,:]
    fx[-1, :] = -Px[-2,:]

    fy = Py - np.concatenate((np.array([Py[:, 0]]).transpose(), Py[:, 0:len(Py)-1]), axis=1)
    fy[:, 0] = Py[:, 0]
    fy[:, -1] = -Py[:, -2]

    return arr_sum(fx, fy)

def chambolleProjection(f, f_ref, iterations = 1000, mi = 100, tau = 0.25, tol = 1e-5):

    n = 1
    xi = np.array([np.zeros(f.shape), np.zeros(f.shape)])
    x1 = np.zeros(f.shape)
    x2 = np.zeros(f.shape)
    x_best = np.zeros(f.shape)

    rms_min_A = []
    rms_min = 1.0
    it_min = 0    

    # start_time = time.time()
    while n - it_min < 100:
        gdv = np.array(gradient2D(divergence2D(xi) - f/mi))
        d = np.sqrt(np.power(gdv[0], 2) + np.power(gdv[1], 2))
        d = np.tile( d, [2, 1, 1] )
        xi = np.divide(xi + tau * gdv, 1 + tau * d)

        x2 = mi * divergence2D(xi)
        
        diff = x2 - f_ref
        rms_n = np.sqrt(np.var(diff.flatten()))
        
        if len(rms_min_A) < 100:
            rms_min_A.append(rms_min)
        else:
            rms_min_A.pop(0)
            rms_min_A.append(rms_min)

        if rms_n < rms_min:
            rms_diff = rms_min_A[0] - rms_min_A[-1]
            rms_local_diff = rms_min - rms_n

            if (rms_diff < 10 * tol):
                if (rms_local_diff < tol):
                    rms_min = rms_n
                    it_min = n
                    break

            rms_min = rms_n
            it_min = n

        x1 = x2
        n = n + 1

    x_best = x2


    # plt.subplot(2, 2, 1)
    # plt.title("Input For VID")
    # plt.imshow(f)
    # plt.subplot(2, 2, 2)
    # plt.title("Output of Chambolle Projection with smallest error")
    # plt.imshow(x_best)
    # plt.subplot(2, 2, 3)
    # plt.title("Texture function")
    # plt.imshow(f_ref)
    # plt.subplot(2, 2, 3)
    # plt.title("Input - Output")
    # plt.imshow(f - x_best)
    # plt.subplot(2, 2, 4)
    # plt.title("Background function")
    # plt.imshow(bg_ref)
    # plt.show()

    return [x_best, it_min, rms_min]