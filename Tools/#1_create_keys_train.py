# Import necessary libraries
import multiprocessing  # is used to speed up the key generation process by utilizing multiple CPU cores.
import os  # module provides a way of using operating system dependent functionality like reading or writing to a file.
import tqdm  # tqdm is a library that provides a progress bar for loops and tasks in the notebook.
from _1_keygen_json import KeyGen  # _1_keygen_json is a custom module for generating encryption keys. KeyGen is a class from this module used specifically for creating keys.

# Configuration parameters
keys_per_file = 100_000
num_key_files = 20


NUMBER_CORS = 16  # Number of CPU cores to use for multiprocessing

# Define directory paths for storing generated data
current_directory = os.getcwd()
WORKING_DIR = os.path.join(current_directory, "..")
PATH_DATA = os.path.join(WORKING_DIR, "Data")
PATH_KEYS = os.path.join(PATH_DATA, "1_keys_train")



# Ensure the existence of the directory to store the keys
os.makedirs(PATH_KEYS, exist_ok=True)

# Begin generation of random keys
key_generator = KeyGen(count=int(keys_per_file))

# Change the working directory to where keys will be stored
os.chdir(PATH_KEYS)
# Generate lug settings with varying overlaps
for i in range(12, 13):
    print(f"Generating lug settings with 1 to {i} overlap")
    key_generator.path = f"overlaps_1-{str(i)}/"
    try:
        os.mkdir(key_generator.path)
    except FileExistsError as err:
        pass
    key_generator.min_overlaps = 1
    key_generator.max_overlaps = i
    filenames = list(str(j).zfill(len(str(num_key_files - 1))) + f'_lugs_1-{str(i)}.json'
                     for j in range(num_key_files))

# Use multiprocessing to generate keys for the specified filenames
with multiprocessing.Pool(NUMBER_CORS) as pool:
        for _ in tqdm.tqdm(pool.imap(key_generator.keygen_json_lugs, filenames), total=num_key_files):
            pass

os.chdir(PATH_KEYS)
i=12 
os.system(f"tar -zcvf overlaps_1-{str(i)}.tar.gz overlaps_1-{str(i)}")
os.system(f"rm -r -f overlaps_1-{str(i)}")

# Preparing for pin generation
os.chdir(PATH_KEYS)

key_generator = KeyGen(count=int(keys_per_file), path="")

filenames = []

# Create directories for pins if it doesn't exist
for i in range(10):
    try:
        os.mkdir(PATH_KEYS + f"/pins{i}/")
    except FileExistsError as err:
        pass
    
    # Generate filenames for the pin files
    filenames += list(PATH_KEYS + f"/pins{i}/" + str(j).zfill(len(str(num_key_files - 1))) + '_pins.json' for j in range(num_key_files))

# Generate pin files using multiprocessing for improved performance
with multiprocessing.Pool(200) as pool:
    for _ in tqdm.tqdm(pool.imap(key_generator.keygen_json_pins, filenames), total=num_key_files):
        pass

#Compress the folders containing the pins and remove the non-compressed ones
os.chdir(PATH_KEYS)
for x in [f"tar -zcvf pins{str(i)}.tar.gz pins{i}" for i in range(10)]:
    os.system(x)

for x in [f"rm -r -f pins{i}" for i in range(10)]:
    os.system(x)