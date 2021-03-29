import time
import math
import numpy as np
import matplotlib.pyplot as plt

from InterferogramGenerator import InterferogramGenerator, InterferogramFromRandomPolynomials
from generateRandomPolynomial import generateRandomPolynomial

def InterGen(folder, imSize = 512, numberOfImages = 1000):
    InGen = InterferogramFromRandomPolynomials(imSize)
    InGen.setFrequencyBoundaries(25, 50)
    InGen.setOrientationBoundaries(0, math.pi)
    InGen.generateALODI(10, 10, numberOfImages, folder)


if __name__ == "__main__":
    start_time = time.time()
    InterGen('C:\\Users\\koste\\Documents\\Python-Source-Codes\\InterGen\\Results\\', 512, 10000)
    print("Execution time: %.2f sec" % (time.time() - start_time))