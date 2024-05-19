import os
import multiprocessing
import random
import time

import numpy as np
import json

# Configuration parameters
NUMBER_OF_OVERLAPS = "1-12"  # Specifies the range of overlaps considered in this dataset


current_directory = os.getcwd()
WORKING_DIR = os.path.join(current_directory, "..")

PATH_DATA = os.path.join(WORKING_DIR, "Data") 

PATH_CIPHERTEXTS = os.path.join(PATH_DATA, "2_ciphertexts_train")
PATH_CIPHERTEXTS = os.path.join(PATH_CIPHERTEXTS, NUMBER_OF_OVERLAPS)

PATH_TRAINING_DATA = os.path.join(PATH_DATA, "3_data_npy_train")

# Ensure the directory for storing training data in NumPy format exists
os.makedirs(PATH_TRAINING_DATA, exist_ok=True)
PATH_TRAINING_DATA = os.path.join(PATH_TRAINING_DATA, NUMBER_OF_OVERLAPS)
os.makedirs(PATH_TRAINING_DATA, exist_ok=True)

#NUMBER_CORS = multiprocessing.cpu_count()
NUMBER_CORS = 50

print(PATH_DATA)
print(WORKING_DIR)

length=500 # Define the length of sequences to be processed

def load_data(file):
    # Define the possible wheel settings as strings
    wheels = ["ABCDEFGHIJKLMNOPQRSTUVWXYZ",
              "ABCDEFGHIJKLMNOPQRSTUVXYZ",
              "ABCDEFGHIJKLMNOPQRSTUVX",
              "ABCDEFGHIJKLMNOPQRSTU",
              "ABCDEFGHIJKLMNOPQRS",
              "ABCDEFGHIJKLMNOPQ"]


    print(f"loading: {file}")
    with open (file, 'r') as infile:
        data = json.load(infile)


    x_temp = []
    y_temp = []
    
    # Process each sequence in the data
    for i in range(len(data)):
        s=data[i][3][:length]
        x_temp.append([ord(n) for n in s])
        new_wheel_data = []
        for j in range(6):
            new_wheel_data += [1 if a in data[i][4][j] else 0 for a in wheels[j]]
        y_temp.append(new_wheel_data)
    
    # Convert the lists to NumPy arrays for use in machine learning models
    x = np.array(x_temp, dtype='ubyte')

    y = np.array(y_temp, dtype='ubyte')
    
    # Extract file name and overlaps information for saving
    file= file.split('/')[-1]
    NUMBER_OF_OVERLAPS = file.split('_')[2].split('.')[0]
    file = file.split('_')[0]

     # Save the processed data as NumPy arrays
    np.save(f"{PATH_TRAINING_DATA  + '/'}{file}_x_{length}_{NUMBER_OF_OVERLAPS}_.npy", x)
    np.save(f"{PATH_TRAINING_DATA  + '/'}{file}_y_ALL_{NUMBER_OF_OVERLAPS}_.npy", y)

filelist = [PATH_CIPHERTEXTS  + '/' + file for file in os.listdir(PATH_CIPHERTEXTS + '/') if "_cipher" in file]

os.listdir(PATH_CIPHERTEXTS + '/')

print (PATH_CIPHERTEXTS )

# Use multiprocessing to process files in parallel for efficiency 
with multiprocessing.Pool(NUMBER_CORS) as pool:
    for _ in pool.imap(load_data, filelist):
        pass
# Change working directory to the training data directory
os.chdir(PATH_TRAINING_DATA)
x_files = [file for file in os.listdir(PATH_TRAINING_DATA) if 'x' in file]
y_files = [file for file in os.listdir(PATH_TRAINING_DATA) if 'y' in file]
def test1():
    x = np.load(f"3_training-data_2/{random.choice(x_files)}")
    y = np.load(f"3_training-data_2/{random.choice(y_files)}")

x_files = [file for file in os.listdir(PATH_TRAINING_DATA) if 'x' in file]
y_files = [file for file in os.listdir(PATH_TRAINING_DATA) if 'y' in file]

# Functions for testing and validating the generated datasets
def test1():
    x = np.load(f"{random.choice(x_files)}")
    x = np.array(x, dtype='float32')
    x = np.subtract(x,65)
    x = np.divide(x , 25)
    

    y = np.load(f"{random.choice(y_files)}")
    y = np.array(y, dtype='float32')
    print(x)
    print(y)

#test1()