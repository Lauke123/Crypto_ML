import os
import random
from tqdm import tqdm
import numpy as np
import json
import multiprocessing

from ._1_keygen_json import inflate_lugs


def calculate_lug_positions(inflated_lugs):
    positions = dict.fromkeys([0, 1, 2, 3, 4, 5, 6], 0)
    for bar in inflated_lugs:
        lug_position =  [int(pos) for pos in bar.split('-')]
        positions[lug_position[0]] += 1
        positions[lug_position[1]] += 1
    keys = positions.keys()
    lug_positions =  [positions[key] for key in sorted(keys)]
    return lug_positions


def load_data(data) -> None:
    file = data[0]
    npy_data_path = data[1]
    # Define the length of sequences to be processed
    length=500
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
    y_temp_lugs = []

    # Process each sequence in the data
    for i in tqdm(range(len(data))):
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
        lug_values = calculate_lug_positions(inflated_lugs)
        y_temp_lugs.append(lug_values)
    
    # Convert the lists to NumPy arrays for use in machine learning models
    x = np.array(x_temp, dtype='ubyte')

    y = np.array(y_temp, dtype='ubyte')

    y_lugs = np.array(y_temp_lugs, dtype='ubyte')
    
    # Extract file name and overlaps information for saving
    file= file.split('/')[-1]
    number_of_overlaps = file.split('_')[2].split('.')[0]
    file = file.split('_')[0]

     # Save the processed data as NumPy arrays
    np.save(f"{npy_data_path  + '/'}{file}_x_{length}_{number_of_overlaps}_.npy", x)
    np.save(f"{npy_data_path  + '/'}{file}_y_ALL_{number_of_overlaps}_.npy", y)
    np.save(f"{npy_data_path  + '/'}{file}_y_lugs_{number_of_overlaps}_.npy", y_lugs)



def create_npy_data_train(cpu_cores: int, output_path: str) -> None:
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
    # Configuration parameters
    number_of_overlaps = "1-12"  # Specifies the range of overlaps considered in this dataset

    output_directory = output_path

    data_directory = os.path.join(output_directory, "Data") 

    ciphertext_directory = os.path.join(data_directory, "2_ciphertexts_train")
    ciphertext_directory = os.path.join(ciphertext_directory, number_of_overlaps)

    npy_data_directory = os.path.join(data_directory, "3_data_npy_train")
    # Ensure the directory for storing training data in NumPy format exists
    os.makedirs(npy_data_directory, exist_ok=True)
    npy_data_directory = os.path.join(npy_data_directory, number_of_overlaps)
    os.makedirs(npy_data_directory, exist_ok=True)

    filelist = [ciphertext_directory  + '/' + file for file in os.listdir(ciphertext_directory + '/') if "_cipher" in file]
    # add the path where the npy_data should be saved to every file in file_list so
    # every child spawned in multiprocessing knows where it should save result
    filelist = [(f, npy_data_directory) for f in filelist]

    os.listdir(ciphertext_directory + '/')

    with multiprocessing.Pool(cpu_cores) as pool:
        for _ in pool.imap(load_data, filelist):
            pass

if __name__ == "__main__":
    create_npy_data_train(os.cpu_count(), "c:/Users/Lukas/Desktop/Bachelor Projekt Cryptoanalysis with ML/code")