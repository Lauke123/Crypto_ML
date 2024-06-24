# multiprocessing is used to speed up the key generation process by utilizing multiple CPU cores.
import multiprocessing

# os and shutil module provides a way of using operating system dependent functionality like reading or writing to a file.
import os
import shutil

# tqdm is a library that provides a progress bar for loops and tasks in the notebook.
import tqdm

# _1_keygen_json is a custom module for generating encryption keys. KeyGen is a class from this module used specifically for creating keys.
from _1_keygen_json import KeyGen


def create_keys_train(cpu_cores: int, output_path: str, keys_per_file: int = 100_000, num_key_files: int = 20) -> None:
    """Create the keys(different settings of the m209 encryption machine) used in the training process.

    Parameters
    ----------
    cpu_cores : int
        amount of cpu cores that should be used during multiprocessing
    output_path : str
        absolute path to the directory the data folder should be created in.
    keys_per_file : int
        the amount of keys that are stored in one file
    num_key_files : int
        the amount of files that hold keys

    Returns
    -------
    None

    """
    # Define directory paths for storing generated data
    output_directory = output_path
    data_directory = os.path.join(output_directory, "Data")
    keys_directory = os.path.join(data_directory, "1_keys_train")

    # Ensure the existence of the directory to store the keys
    os.makedirs(keys_directory, exist_ok=True)

    # Begin generation of random keys
    key_generator = KeyGen(count=int(keys_per_file))

    # Change the working directory to where keys will be stored
    os.chdir(keys_directory)

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
    with multiprocessing.Pool(cpu_cores) as pool:
            for _ in tqdm.tqdm(pool.imap(key_generator.keygen_json_lugs, filenames), total=num_key_files):
                pass

    os.chdir(keys_directory)
    i=12 
    os.system(f"tar -zcvf overlaps_1-{str(i)}.tar.gz overlaps_1-{str(i)}")
    shutil.rmtree(key_generator.path)

    # Preparing for pin generation
    os.chdir(keys_directory)

    key_generator = KeyGen(count=int(keys_per_file), path="")

    filenames = []

    # Create directories for pins if it doesn't exist
    for i in range(10):
        try:
            os.mkdir(keys_directory + f"/pins{i}/")
        except FileExistsError as err:
            pass
        
        # Generate filenames for the pin files
        filenames += list(keys_directory + f"/pins{i}/" + str(j).zfill(len(str(num_key_files - 1))) + '_pins.json' for j in range(num_key_files))

    # Generate pin files using multiprocessing for improved performance
    with multiprocessing.Pool(cpu_cores) as pool:
        for _ in tqdm.tqdm(pool.imap(key_generator.keygen_json_pins, filenames), total=num_key_files):
            pass

    #Compress the folders containing the pins and remove the non-compressed ones
    os.chdir(keys_directory)
    for x in [f"tar -zcvf pins{str(i)}.tar.gz pins{i}" for i in range(10)]:
        os.system(x)

    for pin_folder_path in [f"pins{i}" for i in range(10)]:
        shutil.rmtree(pin_folder_path)

