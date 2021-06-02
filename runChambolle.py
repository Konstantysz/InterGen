import argparse
parser = argparse.ArgumentParser(description="Label images with Chambolle's Projection Algorithm")
parser.add_argument('images_folder', type=str, help="Absolute directory of the dataset")
parser.add_argument('output', type=str, help="Name of the output file")
parser.add_argument('version', type=int, help="Which version of Chambolle algorithm to use, 0 for Stop Criterion, 1 for Reference image")
parser.add_argument('number_of_images', type=int, help="Number of images to process")
parser.add_argument('starting_image', type=int, help="Number of starting image")
args = parser.parse_args()

import numpy as np
import cupy as cp
from PIL import Image
import matplotlib.pyplot as plt
import time

import os, sys, inspect
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],".\\Src\\")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from src.chambolleProjectionGPU import chambolleProjectionGPU, gpuChambolleProjectionStopCriterion
from src.chambolleProjection import chambolleProjection, chambolleProjectionStopCriterion

def runChambolle(images_folder, outputFile, version = 1, number_of_images = 1, starting_image = 0):
    '''
    Function to find number of iterations for Chambolle projection algorithm for specific image. This function works on CPU.
    Results are saved to csv file.

    Source: 
    Cywińska, Maria, Maciej Trusiak, and Krzysztof Patorski. "Automatized fringe pattern preprocessing using unsupervised variational image decomposition." Optics express 27.16 (2019): 22542-22562.

    Parameters
    ----------
    images_folder : str
        folder with images for Chambole input
    outputFile : str
        name of the csv file to put results
    version : int
        version of Chambolle algorithm to use, 0 for Stop Criterio, 1 for Reference image
    number_of_images : int
        number of consecutive images to be processed
    starting_image : int
        id of the image from which algorithm should start 
    '''

    INTERFEROGRAM_PATH = images_folder + "Training\\Interferogram\\"
    FRINGES_PATH = images_folder + "Training\\Fringes\\"
    OUTPUTFILE = ".\\Results\\Labels\\" + outputFile

    csv_file = open(OUTPUTFILE, "a")
    
    try:
        for i in range(starting_image, starting_image + number_of_images):
            filename = str(i) + ".bmp"

            f_path = INTERFEROGRAM_PATH + filename
            f = np.array(Image.open(f_path).convert('L'))
            f = np.divide(f - np.min(f), np.max(f) - np.min(f))
            
            f_ref_path = FRINGES_PATH + filename
            f_ref = np.array(Image.open(f_ref_path).convert('L'))
            f_ref = np.divide(f_ref - np.min(f_ref), np.max(f_ref) - np.min(f_ref))

            start_time = time.time()
            v = np.array([])
            i = 0
            rms = 1.0
            if version == 0:
                [v, i, rms] = chambolleProjection(f, f_ref)
            elif version == 1:
                [v, i, rms] = chambolleProjectionStopCriterion(f)
            stop_time = time.time()

            
            csv_file.write("{},{}\n".format(filename, i))
            print("File: {}".format(filename))
            print("RMS = {}, Itarations: {}".format(rms, i))
            print("--- {} seconds ---".format(stop_time - start_time))

        csv_file.close()
    except KeyboardInterrupt:
        csv_file.close()

def runChambolleGPU(images_folder, outputFile, version = 1, number_of_images = 1, starting_image = 0):
    '''
    Function to find number of iterations for Chambolle projection algorithm for specific image. This function works on GPU.
    Results are saved to csv file.

    Source: 
    Cywińska, Maria, Maciej Trusiak, and Krzysztof Patorski. "Automatized fringe pattern preprocessing using unsupervised variational image decomposition." Optics express 27.16 (2019): 22542-22562.

    Parameters
    ----------
    images_folder : str
        folder with images for Chambole input
    outputFile : str
        name of the csv file to put results
    version : int
        version of Chambolle algorithm to use, 0 for Stop Criterio, 1 for Reference image
    number_of_images : int
        number of consecutive images to be processed
    starting_image : int
        id of the image from which algorithm should start 
    '''

    INTERFEROGRAM_PATH = images_folder + "Training\\Interferogram\\"
    FRINGES_PATH = images_folder + "Training\\Fringes\\"
    OUTPUTFILE = ".\\Results\\Labels\\" + outputFile

    csv_file = open(OUTPUTFILE, "a")
    
    try:
        for i in range(starting_image, starting_image + number_of_images):
            filename = str(i) + ".bmp"

            f_path = INTERFEROGRAM_PATH + filename
            f = cp.array(Image.open(f_path).convert('L'))
            f = cp.divide(f - cp.min(f), cp.max(f) - cp.min(f))
            
            f_ref_path = FRINGES_PATH + filename
            f_ref = cp.array(Image.open(f_ref_path).convert('L'))
            f_ref = cp.divide(f_ref - cp.min(f_ref), cp.max(f_ref) - cp.min(f_ref))

            start_time = time.time()
            v = cp.array([])
            i = 0
            rms = 1.0
            if version == 0:
                [v, i, rms] = chambolleProjectionGPU(f, f_ref)
            elif version == 1:
                [v, i, rms] = gpuChambolleProjectionStopCriterion(f)
            stop_time = time.time()

            csv_file.write("{},{}\n".format(filename, i))
            print("File: {}".format(filename))
            print("RMS = {}, Itarations: {}".format(rms, i))
            print("--- {} seconds ---".format(stop_time - start_time))

        csv_file.close()
    except KeyboardInterrupt:
        csv_file.close()

if __name__ == "__main__":
    cuda = cp.cuda.runtime.getDeviceCount()
    print("Number of CUDA compatible devices:")
    print(cuda)

    if cuda != 0:
        print("Chambolle will run on GPU!")
        runChambolleGPU(args.images_folder, args.output, args.version, args.number_of_images, args.starting_image)
    else:
        print("Chambolle will run on CPU!")
        runChambolle(args.images_folder, args.output, args.version, args.number_of_images, args.starting_image)