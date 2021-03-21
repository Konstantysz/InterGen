import numpy as np

def gauss_n(N):
    amp = 1.0
    sigma = 3.0
    mu = 0.0
    
    x = np.linspace(-1, 1, N)
    y = x
    X, Y = np.meshgrid(x, y)
    
    exponent = ((X - mu)**2 + (Y - mu)**2) / 2*sigma
    val = (amp*np.exp(-exponent))
    
    return val