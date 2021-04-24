import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

from chambolleProjection import chambolleProjection

def runChambolle(results_folder):
    INTERFEROGRAM_PATH = results_folder + "Interferogram\\0.bmp"
    FRINGES_PATH = results_folder + "Fringes\\0.bmp"
    
    f = np.array(Image.open(INTERFEROGRAM_PATH).convert('L'))
    f_ref = np.array(Image.open(FRINGES_PATH).convert('L'))
    
    f = np.divide(f - np.min(f), np.max(f) - np.min(f))
    f_ref = np.divide(f_ref - np.min(f_ref), np.max(f_ref) - np.min(f_ref))

    start_time = time.time()
    [v, i, rms] = chambolleProjection(f, f_ref)
    stop_time = time.time()

    print("RMS = {}, Itarations: {}".format(rms, i))
    print("--- {} seconds ---".format(stop_time - start_time))



if __name__ == "__main__":
    runChambolle('.\\Results\\')