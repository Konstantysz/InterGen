import numpy as np
import math
from PIL import Image, ImageOps

from gauss_n import gauss_n

class InterferogramGenerator:
    
    allInterferograms = []
    
    def __init__(self, N):
        self.__X, self.__Y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
        self.__size = N
        self.__minFrequency = 1
        self.__maxFrequency = 1000
        self.__minOrientationAngle = 0
        self.__maxOrientationAngle = math.pi
        self.__a = gauss_n(self.__size)
        self.__b = 1.0
        self.__n = 0.075*np.random.normal(0.0, 1.0, (self.__size, self.__size))
    
    def setFrequencyBoundaries(self, minF, maxF):
        self.__minFrequency = minF
        self.__maxFrequency = maxF
        
    def setOrientationBoundaries(self, minOA, maxOA):
        self.__minOrientationAngle = minOA
        self.__maxOrientationAngle = maxOA
        
    def setBackgroundFunction(self, background):
        self.__a = background
        
    def setNoiseFunction(self, noise):
        self.__n = noise
        
    def createInterferogram(self, angle, frequency, codedObject):
        phase = math.pi * (math.cos(angle) * self.__X + math.sin(angle) * self.__Y) + codedObject
        I = self.__a + self.__b*np.cos(frequency * phase) + self.__n
        return I
    
    def createMultipleInterferograms(self, codedObject, numOfFrequencies, numOfOrientations):
        freqScalar = (self.__maxFrequency - self.__minFrequency) / numOfFrequencies
        angleScalar = (self.__maxOrientationAngle - self.__minOrientationAngle) / numOfOrientations
        
        for i in range(1, numOfFrequencies):
            for j in range(numOfOrientations+1):
                self.allInterferograms.append(self.createInterferogram(angleScalar * j, freqScalar * i, codedObject))
                
    def saveInterferograms(self):
        for i in range(len(self.allInterferograms)):
            rescaled = (255.0 / self.allInterferograms[i].max() * (self.allInterferograms[i] - self.allInterferograms[i].min())).astype(np.uint8)
            img = Image.fromarray(rescaled)
            filename = 'C:\\Users\\koste\\Documents\\Python-Source-Codes\\InterGen\\Results\\' + str(i) + '.bmp'
            img.convert('RGB').save(filename)


class InterferogramFromRandomPolynomials(InterferogramGenerator):
    s = 0