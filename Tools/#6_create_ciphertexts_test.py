#!/usr/bin/env python
# coding: utf-8



import os
import random
import multiprocessing
import sys
from _3_encrypt import Encrypt




# Define the working directory paths for storing keys, lugs, pins, and ciphertexts

current_directory = os.getcwd()
WORKING_DIR = os.path.join(current_directory, "..")
PATH_DATA = os.path.join(WORKING_DIR, "Data") 
PATH_KEYS = os.path.join(PATH_DATA, "1_keys_test")
PATH_LUGS= os.path.join(PATH_KEYS, "lugs_sorted")
PATH_PINS = os.path.join(PATH_KEYS, "pins")

PATH_CIPHERTEXTS = os.path.join(PATH_DATA, "2_ciphertexts_test")
os.makedirs(PATH_CIPHERTEXTS, exist_ok=True)

# Configuration for multiprocessing and sequence size
#NUMBER_CORS = multiprocessing.cpu_count()
NUMBER_CORS = 16
SEQUENCE_SIZE = 500




# List lug and pin files, filtering by file type
lug_files = [f for f in os.listdir(PATH_LUGS) if "Overlaps" in f]
lug_files = [f for f in lug_files if ".json" in f]
pin_files = [f for f in os.listdir(PATH_PINS) if "pins" in f]
pin_files = [f for f in pin_files if ".json" in f]
#print (lug_files)
#print (pin_files)




# Copy lug and pin files to the ciphertexts directory for processing
os.chdir(PATH_CIPHERTEXTS)
for file in lug_files:
  #  print (file)
    os.system(f"cp {PATH_LUGS+'/'+ file} {PATH_CIPHERTEXTS+'/'}")

for file in pin_files:
  #  print (file)
    os.system(f"cp {PATH_PINS+'/'+file} {PATH_CIPHERTEXTS+'/'}")




import re
os.chdir(PATH_CIPHERTEXTS)

# Shuffle the file lists again for encryption processing

random.shuffle(lug_files)
random.shuffle(pin_files)

# Pair each lug file with a pin file
files = list(zip(lug_files, pin_files))
print (files)

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
    filename, i = x
    m209.load_keys(filename, filetype="pins_n_lugs")
    m209.set_int_msg_ind("AAAAAA")
    a, b = extract_two_integers(f"{filename}")
    
    # Generate a filename based on 'i' with leading zeros
    new_filename = f"{str(i).zfill(len(str(len(files)-1)))}"
    
    # Encrypt the data and save it to a file
    m209.encrypt(new_filename)
    
    # New desired filename format
    desired_filename = f"NS={a}_OV={b}_cipher.json"
    
    # Assuming the encrypted file is saved as new_filename + '_cipher.json' in the working directory
    # Rename the file
    os.rename(f"{new_filename}_cipher.json", desired_filename)
    print(f"File renamed to {desired_filename}")
    




# Execute the encryption and file renaming in parallel using multiprocessing 
with multiprocessing.Pool(NUMBER_CORS) as pool:
    for _ in pool.imap(generate_data, [(files[i], i) for i in range(len(files))]):
        pass




# Clean up the ciphertext directory by removing the original lug and pin files
os.chdir(PATH_CIPHERTEXTS)
for x in [f"rm -r -f {i}" for i in os.listdir() if "pins" in i]:
    os.system(x)

for x in [f"rm -r -f {i}" for i in os.listdir() if "Overlaps" in i]:
    os.system(x)






