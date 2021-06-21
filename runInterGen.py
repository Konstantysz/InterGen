import argparse
parser = argparse.ArgumentParser(description="Generate interferogram images")
parser.add_argument('results_folder', type=str, help="Absolute directory of the folder where to put the results")
args = parser.parse_args()

import time
import json

from src.InterferogramGenerator import InterferogramFromRandomPolynomials

def InterGen(results_folder, settings_filename):
    settings_file = open(settings_filename)
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
    InGen.generateALODI2(
        data['Frequencies']['Frequencies_Per_Object'], 
        data['Number_Of_Images'], 
        results_folder
        )

if __name__ == "__main__":
    start_time = time.time()
    
    InterGen(args.results_folder, "launchInterGen.json")
    print("Execution time: %.2f sec" % (time.time() - start_time))
