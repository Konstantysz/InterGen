import time
import numpy as np
import matplotlib.pyplot as plt
import json

from InterferogramGenerator import InterferogramGenerator, InterferogramFromRandomPolynomials
from generateRandomPolynomial import generateRandomPolynomial

def InterGen(results_folder):
    settings_file = open('settings.json',)
    data = json.load(settings_file)
    settings_file.close()
    
    InGen = InterferogramFromRandomPolynomials(data['Image_Size'])
    InGen.setFrequencyBoundaries(
        data['Frequencies']['Min_Frequency'], 
        data['Frequencies']['Max_Frequency']
        )
    InGen.setOrientationBoundaries(
        data['Orientation']['Min_Orientation_Angle'], 
        data['Orientation']['Max_Orientation_Angle']
        )
    InGen.generateALODI(
        data['Frequencies']['Frequencies_Per_Object'], 
        data['Orientation']['Orientations_Per_Object'], 
        data['Number_Of_Images'], 
        results_folder
        )

if __name__ == "__main__":
    start_time = time.time()
    InterGen('.\\Results\\')
    print("Execution time: %.2f sec" % (time.time() - start_time))
