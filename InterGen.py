import math
import numpy as np
import matplotlib.pyplot as plt

from InterferogramGenerator import InterferogramGenerator
from generateRandomPolynomial import generateRandomPolynomial

def InterGen():
    imSize = 512
    InGen = InterferogramGenerator(imSize)

    phaseObject = generateRandomPolynomial(imSize, 7)

    InGen.setFrequencyBoundaries(25, 50)
    InGen.setOrientationBoundaries(0, math.pi)
    InGen.createMultipleInterferograms(phaseObject, 10, 10)
    InGen.saveInterferograms()

    plt.imshow(InGen.allInterferograms[42])
    plt.show()

if __name__ == "__main__":
    InterGen()