#!/usr/bin/env python
# coding: utf-8



import os
import multiprocessing
import random
import time

import numpy as np
import json
import re

length=500

def load_data(x):

    file, path_testing_data  = x

    wheels = ["ABCDEFGHIJKLMNOPQRSTUVWXYZ",
              "ABCDEFGHIJKLMNOPQRSTUVXYZ",
              "ABCDEFGHIJKLMNOPQRSTUVX",
              "ABCDEFGHIJKLMNOPQRSTU",
              "ABCDEFGHIJKLMNOPQRS",
              "ABCDEFGHIJKLMNOPQ"]


 #   print(f"loading: {file}")
    with open (file, 'r') as infile:
        data = json.load(infile)


    x_temp = []
    y_temp = []

# Process each item in the data list
    for i in range(len(data)):
        s=data[i][3][:length]
        x_temp.append([ord(n) for n in s])
        new_wheel_data = []
        for j in range(6):
            new_wheel_data += [1 if a in data[i][4][j] else 0 for a in wheels[j]]
        y_temp.append(new_wheel_data)

# Convert the lists to numpy arrays with type 'ubyte' for efficient storage
    x = np.array(x_temp, dtype='ubyte')
    y = np.array(y_temp, dtype='ubyte')
    
    # Extract the non-shared lug count and overlap count from the filename using regex
    match = re.search(r"NS=(\d+)_OV=(\d+)_cipher.json", file)
    if not match:
        print(f"Filename {file} does not match expected pattern.")
        return
    n, o = match.groups()
    

    np.save(f"{path_testing_data  + '/'}x_{length}-non-shared-lugs{n}-overlaps{o}.npy", x)
    print (f"{path_testing_data + '/'}x_{length}-non-shared-lugs{n}-overlaps{o}.npy")
    np.save(f"{path_testing_data  + '/'}y_{length}-non-shared-lugs{n}-overlaps{o}.npy", y)

if __name__ == "__main__":
    # Configuration parameters
    NUMBER_OF_OVERLAPS = "1-12"

    # Setting up the directory paths
    current_directory = os.getcwd()
    WORKING_DIR = os.path.join(current_directory, "..")

    PATH_DATA = os.path.join(WORKING_DIR, "Data") 

    PATH_CIPHERTEXTS = os.path.join(PATH_DATA, "2_ciphertexts_test")

    PATH_TESTING_DATA = os.path.join(PATH_DATA, "3_data_npy_test")
    os.makedirs(PATH_TESTING_DATA, exist_ok=True)

    WHEEL = "Wheel1" # In this example working just with Wheel 1

    PATH_TESTING_DATA = os.path.join(PATH_TESTING_DATA, WHEEL)
    os.makedirs(PATH_TESTING_DATA, exist_ok=True)


    #NUMBER_CORS = multiprocessing.cpu_count()
    NUMBER_CORS = os.cpu_count()

    

    os.chdir(PATH_CIPHERTEXTS)
    # Compile a list of files to be processed
    filelist = [(PATH_CIPHERTEXTS + '/'  + file, PATH_TESTING_DATA) for file in os.listdir(PATH_CIPHERTEXTS + '/' ) if "_cipher" in file]
    #print (filelist)




    # Use multiprocessing to process files in parallel for efficiency
    with multiprocessing.Pool(NUMBER_CORS) as pool:
        for _ in pool.imap(load_data, filelist):
            pass
