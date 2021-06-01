import cupy as cp

def gradient2DGPU(mat):
    '''
    Function to calculate gradient of the 2D square matrix. Works on GPU.

    Copyright (c) Gabriel Peyre

    Parameters
    ----------
    mat : cupy.ndarray
        matrix to calculate gradient
        
    Returns:
    ----------
    [fx, fy] : cupy.ndarray
        gradient of `mat`
    '''
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
    '''
    Function to calculate divergence of the 2D square matrix. Works on GPU.

    Copyright (c) Gabriel Peyre

    Parameters
    ----------
    mat : cupy.ndarray
        matrix to calculate divergence

    Returns:
    ----------
    fx + fy : cupy.ndarray
        divergence of `mat`
    '''
    Px = mat[0]
    Py = mat[1]

    fx = Px - cp.concatenate((cp.array([Px[0, :]]), Px[0:len(Px)-1, :]), axis=0)
    fx[0, :] = Px[0,:]
    fx[-1, :] = -Px[-2,:]

    fy = Py - cp.concatenate((cp.array([Py[:, 0]]).transpose(), Py[:, 0:len(Py)-1]), axis=1)
    fy[:, 0] = Py[:, 0]
    fy[:, -1] = -Py[:, -2]

    return fx + fy

def chambolleProjectionGPU(f, f_ref, mi = 100, tau = 0.25, tol = 1e-5):
    '''
    The 2D case of Chambolle projection algorithm. This version uses reference image.

    Source
    -------
    Cywińska, Maria, Maciej Trusiak, and Krzysztof Patorski. 
    "Automatized fringe pattern preprocessing using unsupervised variational image decomposition." Optics express 27.16 (2019): 22542-22562.

    Parameters
    ----------
    f : cupy.ndarray
        image which is input for Chambolle
    f_ref : cupy.ndarray
        image og input but perfectly without background function
    mi : float
        regularization parameter that defines the separation of the energy between the fringes and noise components
    tau : float
        Chambolle projection step value
    tol : float
        error tolerance when algorithm should stop its work

    Returns
    -------
    x_best : numpy.ndarray
        image with filtered background function
    it_min : int
        number of iterations that was needed to reach result image
    rms_min : float
        error of the result image
    '''
    n = 1
    xi = cp.array([cp.zeros(f.shape), cp.zeros(f.shape)])
    x1 = cp.zeros(f.shape)
    x2 = cp.zeros(f.shape)
    x_best = cp.zeros(f.shape)

    rms_min_A = []
    rms_min = 1.0
    it_min = 0    

    for _ in iter(int, 1):
        
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
        
        if n - it_min >= 100:
            break

        pass

    x_best = x2

    return [x_best, it_min, rms_min]

def gpuChambolleProjectionStopCriterion(f, mi = 100, tau = 0.25, tol = 1e-5):
    '''
    The 2D case of Chambolle projection algorithm. This version uses stop criterion.

    Source
    -------
    Cywińska, Maria, Maciej Trusiak, and Krzysztof Patorski. 
    "Automatized fringe pattern preprocessing using unsupervised variational image decomposition." Optics express 27.16 (2019): 22542-22562.

    Parameters
    ----------
    f : cupy.ndarray
        image which is input for Chambolle
    mi : float
        regularization parameter that defines the separation of the energy between the fringes and noise components
    tau : float
        Chambolle projection step value
    tol : float
        error tolerance when algorithm should stop its work

    Returns
    -------
    x2 : numpy.ndarray
        image with filtered background function
    n : int
        number of iterations that was needed to reach result image
    g_err : float
        error of the result image
    '''
    n = 1
    xi = cp.array([cp.zeros(f.shape), cp.zeros(f.shape)])
    x1 = cp.zeros(f.shape)
    x2 = cp.zeros(f.shape)
    cp.cuda.Stream.null.synchronize()

    err_n = 0
    err = []
    pp = []
    pr = 1

    for _ in iter(int, 1):
        
        gdv = cp.array(gradient2DGPU(divergence2DGPU(xi) - f/mi))
        d = cp.sqrt(cp.power(gdv[0], 2) + cp.power(gdv[1], 2))
        d = cp.tile( d, [2, 1, 1] )
        xi = cp.divide(xi + tau * gdv, 1 + tau * d)

        # Reconstruction
        x2 = mi * divergence2DGPU(xi)
        # Tolerance
        num1 = cp.linalg.norm(x2 - x1, 2)
        num2 = cp.linalg.norm(f, 2)
        err.append(num1 / num2)
        
        g_err = cp.abs((err_n - err[n-1])/2)
        err_n = err[n-1]
        pp.append(g_err/err[0])
        pr = pp[n-1]
        
        x1 = x2
        n = n + 1
        
        if pr < tol:
            break

    return [x2, n, g_err]