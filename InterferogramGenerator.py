import numpy as np
import math
from PIL import Image, ImageOps

from gauss_n import gauss_n
from generateRandomPolynomial import generateRandomPolynomial

class InterferogramGenerator:
    
    allInterferograms = []
    
    def __init__(self, N):
        self._X, self._Y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
        self._size = N
        self._minFrequency = 1
        self._maxFrequency = 1000
        self._minOrientationAngle = 0
        self._maxOrientationAngle = math.pi
        self._a = gauss_n(self._size)
        self._b = 1.0
        self._n = 0.075*np.random.normal(0.0, 1.0, (self._size, self._size))
    
    def setFrequencyBoundaries(self, minF, maxF):
        self._minFrequency = minF
        self._maxFrequency = maxF
        
    def setOrientationBoundaries(self, minOA, maxOA):
        self._minOrientationAngle = minOA
        self._maxOrientationAngle = maxOA
        
    def setBackgroundFunction(self, background):
        self._a = background
        
    def setNoiseFunction(self, noise):
        self._n = noise
        
    def createInterferogram(self, angle, frequency, codedObject):
        phase = math.pi * (math.cos(angle) * self._X + math.sin(angle) * self._Y) + codedObject
        I = self._a + self._b*np.cos(frequency * phase) + self._n
        return I
    
    def createMultipleInterferograms(self, codedObject, numOfFrequencies, numOfOrientations):
        freqScalar = (self._maxFrequency - self._minFrequency) / numOfFrequencies
        angleScalar = (self._maxOrientationAngle - self._minOrientationAngle) / numOfOrientations
        
        for i in range(1, numOfFrequencies):
            for j in range(numOfOrientations+1):
                self.allInterferograms.append(self.createInterferogram(angleScalar * j, freqScalar * i, codedObject))
                
    def saveInterferograms(self, folder, startNum = 0):
        for i in range(len(self.allInterferograms)):
            rescaled = (255.0 / self.allInterferograms[i].max() * (self.allInterferograms[i] - self.allInterferograms[i].min())).astype(np.float64)
            img = Image.fromarray(rescaled)
            filename = folder + str(i + startNum) + '.bmp'
            img.convert('RGB').save(filename)


class InterferogramFromRandomPolynomials(InterferogramGenerator):
    
    ### Method to generate A Lot Of Different Interferograms
    def generateALODI(self, numOfFrequencies, numOfOrientations, quantity, folder):
        nbOfObjects = quantity / (numOfFrequencies * numOfOrientations)
        freqScalar = (self._maxFrequency - self._minFrequency) / numOfFrequencies
        angleScalar = (self._maxOrientationAngle - self._minOrientationAngle) / numOfOrientations
        
        for i in range(int(nbOfObjects)):
            degree = np.random.randint(5) + 5
            obj = generateRandomPolynomial(self._size, degree)

            for j in range(1, numOfFrequencies + 1):
                for k in range(numOfOrientations):
                    self.allInterferograms.append(self.createInterferogram(angleScalar * k, freqScalar * j, obj))

            self.saveInterferograms(folder, int(numOfOrientations*numOfFrequencies*i))
            self.allInterferograms = []

            print("Progress: " + str(i + 1) + "/" + str(int(nbOfObjects)))


