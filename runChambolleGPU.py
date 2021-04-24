import numpy as np
import cupy as cp
from PIL import Image
import matplotlib.pyplot as plt
import time

from chambolleProjectionGPU import chambolleProjectionGPU

def runChambolleGPU(results_folder):
    INTERFEROGRAM_PATH = results_folder + "Interferogram\\8.bmp"
    FRINGES_PATH = results_folder + "Fringes\\8.bmp"
    
    f = cp.array(Image.open(INTERFEROGRAM_PATH).convert('L'))
    f_ref = cp.array(Image.open(FRINGES_PATH).convert('L'))
    
    f = cp.divide(f - cp.min(f), cp.max(f) - cp.min(f))
    f_ref = cp.divide(f_ref - cp.min(f_ref), cp.max(f_ref) - cp.min(f_ref))

    start_time = time.time()
    [v, i, rms] = chambolleProjectionGPU(f, f_ref)
    stop_time = time.time()

    print("RMS = {}, Itarations: {}".format(rms, i))
    print("--- {} seconds ---".format(stop_time - start_time))



if __name__ == "__main__":
    runChambolleGPU('.\\Results\\')