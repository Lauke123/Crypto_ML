import multiprocessing
import os
import random
import re
import shutil

from ._3_encrypt import Encrypt

SEQUENCE_SIZE = 500

# Initialize the encryption object with specified configurations
m209 = Encrypt(destin_path="", count_a=SEQUENCE_SIZE,
               append_ciphertext=False,
               append_plaintext=False,
               append_keystream=True,
               append_lugs=True,
               append_pins=True)

def extract_two_integers(input_string):
    # Define a regex pattern for integers
    pattern = r'\d+'
    # Find all matches of the pattern in the string
    matches = re.findall(pattern, input_string)
    # Convert the first two matches to integers, if any
    integers = [int(match) for match in matches][:2]
    return integers


def generate_data(x):
    filename, i, files_count = x
    m209.load_keys(filename, filetype="pins_n_lugs")
    m209.set_int_msg_ind("AAAAAA")
    a, b = extract_two_integers(f"{filename}")
    
    # Generate a filename based on 'i' with leading zeros
    new_filename = f"{str(i).zfill(len(str(files_count-1)))}"
    
    # Encrypt the data and save it to a file
    m209.encrypt(new_filename)
    
    # New desired filename format
    desired_filename = f"NS={a}_OV={b}_cipher.json"
    
    # Assuming the encrypted file is saved as new_filename + '_cipher.json' in the working directory
    # Rename the file
    os.rename(f"{new_filename}_cipher.json", desired_filename)
    print(f"File renamed to {desired_filename}")
    

# guard so child processes do not execute this part (important if executed on windows machines)
def create_ciphertext_test(cpu_cores: int, output_path: str) -> None:
    """Create the ciphertext for testing according to keys stored in the previous step.

    Parameters
    ----------
    cpu_cores : int
        amount of cpu cores that should be used during multiprocessing
    output_path : str
        absolute path to the directory the data folder from the previous step is inside

    Returns
    -------
    None

    """
    # Define the working directory paths for storing keys, lugs, pins, and ciphertexts

    output_directory = output_path
    data_directory = os.path.join(output_directory, "Data") 
    keys_directory = os.path.join(data_directory, "1_keys_test")
    lugs_directory= os.path.join(keys_directory, "lugs_sorted")
    pins_directory = os.path.join(keys_directory, "pins")

    ciphertext_directory = os.path.join(data_directory, "2_ciphertexts_test")
    os.makedirs(ciphertext_directory, exist_ok=True)

    # List lug and pin files, filtering by file type
    lug_files = [f for f in os.listdir(lugs_directory) if "Overlaps" in f]
    lug_files = [f for f in lug_files if ".json" in f]
    pin_files = [f for f in os.listdir(pins_directory) if "pins" in f]
    pin_files = [f for f in pin_files if ".json" in f]

    # Copy lug and pin files to the ciphertexts directory for processing
    os.chdir(ciphertext_directory)
    for file in lug_files:
        shutil.copy(lugs_directory + '/' + file, ciphertext_directory)

    for file in pin_files:
        shutil.copy(pins_directory + '/' + file, ciphertext_directory)

    os.chdir(ciphertext_directory)

    # Shuffle the file lists again for encryption processing
    random.shuffle(lug_files)
    random.shuffle(pin_files)

    # Pair each lug file with a pin file
    files = list(zip(lug_files, pin_files))

    # Execute the encryption and file renaming in parallel using multiprocessing 
    with multiprocessing.Pool(cpu_cores) as pool:
        for _ in pool.imap(generate_data, [(files[i], i, len(files)) for i in range(len(files))]):
            pass

    # Clean up the ciphertext directory by removing the original lug and pin files
    os.chdir(ciphertext_directory)
    for x in [f"{i}" for i in os.listdir() if "pins" in i]:
        os.remove(x)

    for x in [f"{i}" for i in os.listdir() if "Overlaps" in i]:
        os.remove(x)






