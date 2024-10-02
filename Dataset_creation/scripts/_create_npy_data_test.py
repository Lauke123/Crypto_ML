import json
import multiprocessing
import os
import re
from collections import defaultdict
from itertools import combinations

import numpy as np

from ._1_keygen_json import inflate_lugs


def count_lug_pairs(input_list):
    pair_dict = defaultdict(int)
    
    # Count existing combinations from input_list
    for pair in input_list:
        x, y = pair.split('-')
        sorted_pair = tuple(sorted([x, y]))
        pair_dict[sorted_pair] += 1
    
    # Generate all valid combinations (0-0 allowed, but no other x-x where x != 0)
    valid_pairs = list(combinations(map(str, range(1, 7)), 2)) + [('0', '0')] + [('0', str(i)) for i in range(1, 7)]
    
    # Create the final dictionary and set counts (0 for those not found in the input list)
    full_pair_dict = {pair: pair_dict.get(pair, 0) for pair in valid_pairs}
    
    # Convert to a sorted list of tuples
    sorted_pair_list = sorted(full_pair_dict.items())
    value_pair_list = [pair[1] for pair in sorted_pair_list]
    
    return value_pair_list


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
    y_temp_lugs = []

    # Process each item in the data list
    for i in range(len(data)):
        s=data[i][3][:length]
        x_temp.append([ord(n) for n in s])
        new_wheel_data = []
        for j in range(6):
            new_wheel_data += [1 if a in data[i][4][j] else 0 for a in wheels[j]]
        y_temp.append(new_wheel_data)

        # add the number of lugs to the data that are on either pos 0,1,2,3,4,5,6
        lug_values = []
        lugs = data[i][5]
        inflated_lugs = inflate_lugs(lugs)
        lug_values = count_lug_pairs(inflated_lugs)
        y_temp_lugs.append(lug_values)

    # Convert the lists to numpy arrays with type 'ubyte' for efficient storage
    x = np.array(x_temp, dtype='ubyte')
    y = np.array(y_temp, dtype='ubyte')
    y_lugs = np.array(y_temp_lugs, dtype='ubyte')

    # Extract the non-shared lug count and overlap count from the filename using regex
    match = re.search(r"NS=(\d+)_OV=(\d+)_cipher.json", file)
    if not match:
        print(f"Filename {file} does not match expected pattern.")
        return
    n, o = match.groups()

    np.save(f"{npy_testing_data_directory  + '/'}x_{length}-non-shared-lugs{n}-overlaps{o}.npy", x)
    print (f"{npy_testing_data_directory + '/'}x_{length}-non-shared-lugs{n}-overlaps{o}.npy")
    np.save(f"{npy_testing_data_directory  + '/'}y_{length}-non-shared-lugs{n}-overlaps{o}.npy", y)
    np.save(f"{npy_testing_data_directory  + '/'}y_lugs_{length}-non-shared-lugs{n}-overlaps{o}.npy", y_lugs)

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

if __name__ == "__main__":
    create_npy_data_test(os.cpu_count(), "c:/Users/Lukas/Desktop/Bachelor Projekt Cryptoanalysis with ML/code")