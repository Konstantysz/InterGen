import numpy as np
import cupy as cp
from PIL import Image
import matplotlib.pyplot as plt
import time
import csv

import os, sys, inspect
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],".\\Src\\")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from src.chambolleProjectionGPU import chambolleProjectionGPU, gpuChambolleProjectionStopCriterion

def runChambolleGPU(results_folder):
    INTERFEROGRAM_PATH = results_folder + "Training\\Interferogram\\"
    FRINGES_PATH = results_folder + "Training\\Fringes\\"

    starting_image = 5107
    number_of_images = 393

    csv_file = open(".\\Results\\Labels\\labels_SC_5000_5500.csv", "a")
    
    try:
        for i in range(starting_image, starting_image+number_of_images):
            filename = str(i) + ".bmp"
            f_path = INTERFEROGRAM_PATH + filename
            # f_ref_path = FRINGES_PATH + filename
            
            f = cp.array(Image.open(f_path).convert('L'))
            # f_ref = cp.array(Image.open(f_ref_path).convert('L'))
        
            f = cp.divide(f - cp.min(f), cp.max(f) - cp.min(f))
            # f_ref = cp.divide(f_ref - cp.min(f_ref), cp.max(f_ref) - cp.min(f_ref))

            start_time = time.time()
            # [v, i, rms] = chambolleProjectionGPU(f, f_ref)
            [v, i] = gpuChambolleProjectionStopCriterion(f)
            stop_time = time.time()

            # csv_file.write("{},{},{}\n".format(filename, i, rms))
            # print("File: {}".format(filename))
            # print("RMS = {}, Itarations: {}".format(rms, i))
            # print("--- {} seconds ---".format(stop_time - start_time))
            csv_file.write("{},{}\n".format(filename, i))
            print("File: {}".format(filename))
            print("Itarations: {}".format(i))
            print("--- {} seconds ---".format(stop_time - start_time))

        csv_file.close()
    except KeyboardInterrupt:
        csv_file.close()

if __name__ == "__main__":
    print(cp.cuda.runtime.getDeviceCount())
    runChambolleGPU('C:\\Users\\koste\\Desktop\\Studia\\_STUDIA MAGISTERSKIE\\_PRACA MAGISTERSKA\\uVID\\uVID\\Dataset\\Images\\')