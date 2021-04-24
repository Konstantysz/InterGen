from numba import jit
import time
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

def gradient2DGPU(mat):
    x1 = mat[1:len(mat), :]
    x2 = cp.array([mat[-1, :]])
    x = cp.concatenate((x1, x2), axis = 0)

    fx = x - mat
    y1 = mat[:, 1:len(mat)]
    y2 = cp.array([mat[:, -1]]).transpose()
    y = cp.concatenate((y1, y2), axis = 1)

    fy = y - mat

    return [fx, fy]

def divergence2DGPU(mat):
    Px = mat[0]
    Py = mat[1]

    fx = Px - cp.concatenate((cp.array([Px[0, :]]), Px[0:len(Px)-1, :]), axis=0)
    fx[0, :] = Px[0,:]
    fx[-1, :] = -Px[-2,:]

    fy = Py - cp.concatenate((cp.array([Py[:, 0]]).transpose(), Py[:, 0:len(Py)-1]), axis=1)
    fy[:, 0] = Py[:, 0]
    fy[:, -1] = -Py[:, -2]

    return fx + fy

def chambolleProjectionGPU(f, f_ref, iterations = 1000, mi = 100, tau = 0.25, tol = 1e-5):

    n = 1
    xi = cp.array([cp.zeros(f.shape), cp.zeros(f.shape)])
    x1 = cp.zeros(f.shape)
    x2 = cp.zeros(f.shape)
    x_best = cp.zeros(f.shape)

    rms_min_A = []
    rms_min = 1.0
    it_min = 0    

    while n - it_min < 100:
        gdv = cp.array(gradient2DGPU(divergence2DGPU(xi) - f/mi))
        d = cp.sqrt(cp.power(gdv[0], 2) + cp.power(gdv[1], 2))
        d = cp.tile( d, [2, 1, 1] )
        xi = cp.divide(xi + tau * gdv, 1 + tau * d)

        x2 = mi * divergence2DGPU(xi)
        
        diff = x2 - f_ref
        rms_n = cp.sqrt(cp.var(diff.flatten()))

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

    return [x_best, it_min, rms_min]