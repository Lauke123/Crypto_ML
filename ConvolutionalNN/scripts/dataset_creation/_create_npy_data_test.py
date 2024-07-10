import json
import multiprocessing
import os
import re

import numpy as np


def load_data(data):
    length=500
    file, npy_testing_data_directory  = data

    wheels = ["ABCDEFGHIJKLMNOPQRSTUVWXYZ",
              "ABCDEFGHIJKLMNOPQRSTUVXYZ",
              "ABCDEFGHIJKLMNOPQRSTUVX",
              "ABCDEFGHIJKLMNOPQRSTU",
              "ABCDEFGHIJKLMNOPQRS",
              "ABCDEFGHIJKLMNOPQ"]

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

    np.save(f"{npy_testing_data_directory  + '/'}x_{length}-non-shared-lugs{n}-overlaps{o}.npy", x)
    print (f"{npy_testing_data_directory + '/'}x_{length}-non-shared-lugs{n}-overlaps{o}.npy")
    np.save(f"{npy_testing_data_directory  + '/'}y_{length}-non-shared-lugs{n}-overlaps{o}.npy", y)

def create_npy_data_test(cpu_cores: int, output_path: str) -> None:
    """Create the data that is used during the training process.

    Parameters
    ----------
    cpu_cores : int
        amount of cpu cores that should be used during multiprocessing
    output_path : str
        absolute path to the directory the data folder should be created in.

    Returns
    -------
    None

    """
    # Setting up the directory paths
    output_directory = output_path
    data_directory = os.path.join(output_directory, "Data") 
    ciphertext_directory = os.path.join(data_directory, "2_ciphertexts_test")
    npy_testing_data_directory = os.path.join(data_directory, "3_data_npy_test")
    os.makedirs(npy_testing_data_directory, exist_ok=True)

    WHEEL = "Wheel1" # In this example working just with Wheel 1
    npy_testing_data_directory = os.path.join(npy_testing_data_directory, WHEEL)
    os.makedirs(npy_testing_data_directory, exist_ok=True)
    os.chdir(ciphertext_directory)

    # Compile a list of files to be processed
    filelist = [(ciphertext_directory + '/'  + file, npy_testing_data_directory) for file in os.listdir(ciphertext_directory + '/' ) if "_cipher" in file]

    # Use multiprocessing to process files in parallel for efficiency
    with multiprocessing.Pool(cpu_cores) as pool:
        for _ in pool.imap(load_data, filelist):
            pass
