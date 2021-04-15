from numba import jit
import numpy as np
import math
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from gauss_n import gauss_n
from generateRandomPolynomial import generateRandomPolynomial
from generateSphericalObject import generateSphericalObject
from chambolleProjection import chambolleProjection
from normalizeImage import normalizeImage
from progressBar import progressBar

class InterferogramGenerator:
    """
    A class used to generate interferogram

    ...

    Attributes
    ----------
    allInterferograms : numpy.ndarray
        array that stores images of interferograms (Optional to use)
    _X, _Y : numpy.ndarray
        distribution of values from -1 to 1
    _size : int
        size of the interferogram image
    _minFrequency : int
        minimal spatial frequency of fringe pattern
    _maxFrequency : int
        maximal spatial frequency of fringe pattern
    _minOrientationAngle : float
        minimal orientation angle of fringe pattern
    _maxOrientationAngle : float
        maximal orientation angle of fringe pattern
    _a : numpy.ndarray
        function of background of fringe pattern
    _b : numpy.ndarray
        function of amplitude of fringe pattern
    _n : numpy.ndarray
        function of noise of fringe pattern

    Methods
    -------
    setFrequencyBoundaries(minF, maxF)
        Sets range of spatial frequencies of interferogram
    setOrientationBoundaries(minOA, maxOA)
        Sets range of orientations of interferogram
    setBackgroundFunction(background)
        Sets background function of interferogram
    setNoiseFunction(noise)
        Sets noise function of interferogram
    createInterferogram(angle, frequency, codedObject)
        Returns single interferogram image
    createMultipleInterferograms(codedObject, numOfFrequencies, numOfOrientations)
        Creates multiple interferogram images and stores them in `self.allInterferograms`
    saveInterferograms(folder, startNum = 0)
        Saves interferograms stored in `self.allInterferograms`
    """
    allInterferograms = []
    
    def __init__(self, N):
        """
        Parameters
        ----------
        N : int
            Size of the image
        """
        self._X, self._Y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
        self._size = N
        self._minFrequency = 1
        self._maxFrequency = 1000
        self._minOrientationAngle = 0.0
        self._maxOrientationAngle = math.pi
        self._a = gauss_n(self._size)
        self._b = 1.0
        self._n = 0.075*np.random.normal(0.0, 1.0, (self._size, self._size))
    
    def setFrequencyBoundaries(self, minF, maxF):
        '''
        Sets range of spatial frequencies of interferogram

        Parameters
        ----------
        minF : int
            minimal spatial frequency of fringe pattern
        maxF : int
            maximal spatial frequency of fringe pattern
        '''
        self._minFrequency = minF
        self._maxFrequency = maxF
        
    def setOrientationBoundaries(self, minOA, maxOA):
        '''
        Sets range of orientations of interferogram

        Parameters
        ----------
        minOA : float
            minimal orientation angle of fringe pattern
        maxOA : float
            maximal orientation angle of fringe pattern
        '''
        self._minOrientationAngle = minOA
        self._maxOrientationAngle = maxOA
        
    def setBackgroundFunction(self, background):
        '''
        Sets background function of interferogram

        Parameters
        ----------
        background : numpy.ndarray
            function of background of fringe pattern
        '''
        self._a = background
        
    def setNoiseFunction(self, noise):
        '''
        Sets noise function of interferogram

        Parameters
        ----------
        noise : numpy.ndarray
            function of noise of fringe pattern
        '''
        self._n = noise

    def createInterferogram(self, angle, frequency, phaseObject):
        '''
        Returns single interferogram image

        Attributes
        ----------
        angle : float
            orientation angle of fringe pattern
        frequency : int
            spatial frequency of fringe pattern
        phaseObject : numpy.ndarray
            object to be coded in the phase of the interferogram
        '''
        angle = math.pi / 2 * (math.cos(angle) * self._X + math.sin(angle) * self._Y) + phaseObject
        I = self._a + self._b*np.cos(frequency * angle) + self._n
        return I
    
    def createSphericalInterferogram(self, obj):
        '''
        Returns single interferogram image of spherical fringes

        Attributes
        ----------
        obj : ndArray
            shift of sphere in x axis
        '''
        I = self._a + self._b*np.cos(obj) + self._n
        return I
    

    def createMultipleInterferograms(self, phaseObject, numOfFrequencies, numOfOrientations):
        '''
        Creates multiple interferogram images and stores them in `self.allInterferograms`

        Attributes
        ----------
        phaseObject : numpy.ndarray
            object to be coded in the phase of the interferogram
        numOfFrequencies : int
            number of different spatial frequencies of 
            fringe pattern to be used for single coded object
        numOfOrientations : int
            number of different orientations of fringe 
            pattern to be used for single coded object
        '''
        freqScalar = (self._maxFrequency - self._minFrequency) / numOfFrequencies
        angleScalar = (self._maxOrientationAngle - self._minOrientationAngle) / numOfOrientations
        
        for i in range(1, numOfFrequencies):
            for j in range(numOfOrientations+1):
                self.allInterferograms.append(self.createInterferogram(angleScalar * j, freqScalar * i, phaseObject))
                
    def saveInterferograms(self, folder, startNum = 0):
        '''
        Saves interferograms stored in `self.allInterferograms`

        Attributes
        ----------
        folder : str
            path to folder in which images will be saved
        startNum : int
            starting number for numeration of filenames (Default is 0)
        '''
        for i in range(len(self.allInterferograms)):
            rescaled = (255.0 / self.allInterferograms[i].max() * (self.allInterferograms[i] - self.allInterferograms[i].min())).astype(np.float64)
            img = Image.fromarray(rescaled)
            filename = folder + str(i + startNum) + '.bmp'
            img.convert('RGB').save(filename)


class InterferogramFromRandomPolynomials(InterferogramGenerator):
    """
    Class derived from `InterferogramGenerator` used to generate interferograms from random polynomial functions encoded in object and in background function

    ...

    Methods
    -------
    saveInterferogram(image, folder, interferogramNumber)
        Sets range of spatial frequencies of interferogram
    generateALODI(numOfFrequencies, numOfOrientations, quantity, folder)
        Method to generate A Lot Of Different Interferograms
    """
    def saveInterferogram(self, image, folder, interferogramNumber):
        '''
        Saves single interferogram

        Attributes
        ----------
        image : numpy.ndarray
            interferogram to be saved
        folder : str
            path to folder in which image will be saved
        interferogramNumber : int
            number that is going to be filename
        '''
        rescaled = (255.0 / image.max() * (image - image.min())).astype(np.float64)
        img = Image.fromarray(rescaled)
        filename = folder + str(interferogramNumber) + '.bmp'
        img.convert('RGB').save(filename)

    def generateALODI(self, numOfFrequencies, numOfOrientations, quantity, folder):
        '''
        Generates multiple interferograms. Quantity is specified by user. 
        Interferograms are generated with random object in phase and random background function.

        Attributes
        ----------
        numOfFrequencies : int
            number of different frequencies of fringe pattern that single pattern will be generated with
        numOfOrientations : int
            number of different frequencies of fringe pattern that single pattern will be generated with
        quantity : int
            number of all generated interferograms
        folder : str
            path to folder in which image will be saved
        '''
        x = np.linspace(-1, 1, self._size)
        y = np.linspace(-1, 1, self._size)
        X, Y = np.meshgrid(x, y)

        for i in range(quantity):
            objType = np.random.choice(np.arange(3), p=[0.01, 0.04, 0.95])
            '''
            0 -> Prazki liniowe
            1 -> Prazki kolowe
            2 -> Prazki wielomianowe stopnia najwyzej 3
            '''
            if objType == 0:
                obj = generateRandomPolynomial(X, Y, 1)
            elif objType == 2:
                obj = generateRandomPolynomial(X, Y, 3)

            bg = generateRandomPolynomial(X, Y, 10)
            max_abs = max(bg.min(), bg.max(), key=abs)
            bg = bg / max_abs
            self.setBackgroundFunction(bg)
                    
            if objType == 1:
                x0 = np.random.uniform(-0.5, 0.5)
                y0 = np.random.uniform(-0.5, 0.5)
                f = np.random.randint(0.5 * self._minFrequency, 2 * self._maxFrequency)
                h = np.random.randint(-3, 3)

                spObj = generateSphericalObject(X, Y, x0, y0, f, h)
                I = normalizeImage(self.createSphericalInterferogram(spObj))
                refI = normalizeImage(self._b*np.cos(spObj))
                refBG = normalizeImage(bg)

                chambolleProjection(I, refBG)

                self.saveInterferogram(I, folder, i)
            else:
                freq = np.random.randint(self._minFrequency, self._maxFrequency)
                angle = np.random.randint(self._minOrientationAngle, self._maxOrientationAngle)

                I = normalizeImage(self.createInterferogram(angle, freq, obj))
                refI = normalizeImage(
                    self._b*np.cos(freq * (math.pi / 2 * (math.cos(angle) * self._X + math.sin(angle) * self._Y) + obj))
                )
                refBG = normalizeImage(bg)

                chambolleProjection(I, refBG)

                self.saveInterferogram(I, folder, i)

            progressBar(i + 1, quantity, "Interferogram generation progress: ")
