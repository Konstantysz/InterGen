import numpy as np
import cupy as cp
from PIL import Image
import matplotlib.pyplot as plt
import time
import csv

from chambolleProjectionGPU import chambolleProjectionGPU

def runChambolleGPU(results_folder):
    INTERFEROGRAM_PATH = results_folder + "Interferogram\\"
    FRINGES_PATH = results_folder + "Fringes\\"

    starting_image = 9000
    number_of_images = 500

    csv_file = open("labels_itNum_9000_9499.csv", "a")

    for i in range(starting_image, starting_image+number_of_images):
        filename = str(i) + ".bmp"
        f_path = INTERFEROGRAM_PATH + filename
        f_ref_path = FRINGES_PATH + filename
        
        f = cp.array(Image.open(f_path).convert('L'))
        f_ref = cp.array(Image.open(f_ref_path).convert('L'))
    
        f = cp.divide(f - cp.min(f), cp.max(f) - cp.min(f))
        f_ref = cp.divide(f_ref - cp.min(f_ref), cp.max(f_ref) - cp.min(f_ref))

        start_time = time.time()
        [v, i, rms] = chambolleProjectionGPU(f, f_ref)
        stop_time = time.time()

        csv_file.write("{},{},{}\n".format(filename, i, rms))
        print("File: {}".format(filename))
        print("RMS = {}, Itarations: {}".format(rms, i))
        print("--- {} seconds ---".format(stop_time - start_time))

    csv_file.close()


if __name__ == "__main__":
    runChambolleGPU('.\\Results\\')