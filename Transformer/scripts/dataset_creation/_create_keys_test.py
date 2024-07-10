import multiprocessing
import os
import tqdm    
from collections import defaultdict, Counter
from pathlib import Path
import json
import re

from ._1_keygen_json import KeyGen

def create_keys_test(cpu_cores: int, output_path: str, lugs_keys_per_file: int = 1000, lugs_num_key_files: int = 20, 
                     pin_num_key_files: int = 103, pin_keys_per_file: int = 1000,
                     lugs_mixed_keys_per_file: int = 40000, lugs_mixed_num_key_files: int = 200) -> None:
    """Create the keys(different settings of the m209 encryption machine) used in the testing process.

    Parameters
    ----------
    cpu_cores : int
        amount of cpu cores that should be used during multiprocessing
    output_path : str
        absolute path to the directory the data folder should be created in.
    lugs_keys_per_file : int
        the amount of keys that are stored in one file of lug settings that are used to calculate files for a specific amount of overlaps
    lugs_num_key_files : int
        the amount of files that hold keys for specific overlaps
    pin_keys_per_file : int
        the amount of keys for pin setting that are stored in one file
    pin_num_key_files : int
        the amount of files that hold keys for pin settings
    lugs_mixed_keys_per_file : int
        amount of different lugs setting storend in one file, from 1 to max overlap all lug settigs are mixed together
    lugs_mixed_num_key_files : int
        the amount of key files of mixed lugs settings

    Returns
    -------
    None

    """
    # Define directory paths for storing generated data
    output_directory = output_path
    data_directory = os.path.join(output_directory, "Data") 
    keys_directory = os.path.join(data_directory, "1_keys_test")
    lugs_directory= os.path.join(keys_directory, "lugs")
    sorted_lugs_directory= os.path.join(keys_directory, "lugs_sorted")

    # Ensure the existence of the directories
    os.makedirs(keys_directory, exist_ok=True)
    os.makedirs(lugs_directory, exist_ok=True)
    os.makedirs(sorted_lugs_directory, exist_ok=True)

    # Begin generation of random keys
    print("\n\nGenerating random keys")
    key_generator = KeyGen(count=int(lugs_keys_per_file))

    # Change the working directory to where lugs will be stored
    os.chdir(lugs_directory)
    # Generate lug settings with varying overlaps
    for i in range(1, 13):
        print(f"Generating lug settings with {i} overlap")
        if (i == 5): # Increase the number of key files significantly for keys with more overlaps
            lugs_num_key_files=lugs_num_key_files*20

        key_generator.path = f"overlaps_{str(i)}/"
        try:
            os.mkdir(key_generator.path)
        except FileExistsError as err:
            pass # Ignore if the directory already exists
        key_generator.min_overlaps = i
        key_generator.max_overlaps = i
        filenames = list(str(j).zfill(len(str(lugs_num_key_files - 1))) + f'_lugs_{str(i)}.json'
                        for j in range(lugs_num_key_files))

    # Use multiprocessing to generate keys for the specified filenames
        with multiprocessing.Pool(cpu_cores) as pool:
            for _ in tqdm.tqdm(pool.imap(key_generator.keygen_json_lugs, filenames),total=lugs_num_key_files):
                pass

    # Generate lug settings with 1 to max overlap
    os.chdir(lugs_directory)

    key_generator = KeyGen(count=int(lugs_mixed_keys_per_file))
    for i in range(12, 13):
        print(f"Generating lug settings with 1 to {i} overlap")
        key_generator.path = f"overlaps_1-{str(i)}/"
        try:
            os.mkdir(key_generator.path)
        except FileExistsError as err:
            pass
        key_generator.min_overlaps = 1
        key_generator.max_overlaps = i
        filenames = list(str(j).zfill(len(str(lugs_mixed_num_key_files - 1))) + f'_lugs_1-{str(i)}.json'
                        for j in range(lugs_mixed_num_key_files))

        with multiprocessing.Pool(cpu_cores) as pool:
            for _ in tqdm.tqdm(pool.imap(key_generator.keygen_json_lugs, filenames), total=lugs_mixed_num_key_files):
                pass

    # Define functions for extracting and processing lug sequences from generated JSON files
    def extract_values(sequence):
    # Extracts and returns the values from a sequence.
        n_value, k_values_sum = 0, 0
        for match in re.finditer(r'1-(\d)(?:\*(\d+))?', sequence):
            x, k = int(match.group(1)), int(match.group(2)) if match.group(2) else 1
            if x == 0:
                n_value += k
            else:
                k_values_sum += k
        return n_value, k_values_sum

    def read_json_file(json_file):
    #  Reads and returns the content of a JSON file.
        with open(json_file, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading {json_file}")
                return []

    def collect_sequences(start_path):
        #Collects sequences from JSON files within the specified path.
        path = Path(start_path)
        sequences_by_type = defaultdict(list)
        type_counts = Counter()

        # Collect sequences
        for json_file in path.rglob('*.json'):
            sequences = read_json_file(json_file)
            for seq in sequences:
                n, k_sum = extract_values(seq)
                type_key = (n, k_sum)
                if type_counts[type_key] < 1000:  # Only keep tracking if under 1000
                    sequences_by_type[type_key].append(seq)
                    type_counts[type_key] += 1

        # Filter out types with more than 1000 instances
        for type_key, count in list(type_counts.items()):
            if count > 10000:
                del sequences_by_type[type_key]

        return sequences_by_type

    def save_sequences(sequences_by_type, working_folder):
        #Saves the collected sequences to files, organized by their types
        for (n, k_sum), seqs in sequences_by_type.items():
            filename = Path(working_folder) / f"Non-shared={n}_Overlaps={k_sum}.json"
            # Since we're writing at the end, no need to check for existing content
            with open(filename, 'w') as f:
                json.dump(seqs, f)
            print(f"Saved {len(seqs)} sequences to {filename}")

    def main(start_path, working_folder):
        sequences_by_type = collect_sequences(start_path)
        save_sequences(sequences_by_type, working_folder)

    # Adjust 'start_path' and 'working_folder' as needed
    start_path = lugs_directory
    working_folder = sorted_lugs_directory
    main(start_path, working_folder)



    # Preparing for pin generation
    os.chdir(keys_directory)
    key_generator = KeyGen(count=int(pin_keys_per_file), path="")
    filenames = []
    # Create directory for pins if it doesn't exist
    try:
        os.mkdir(keys_directory + f"/pins/")
    except FileExistsError as err:
        pass

    # Generate filenames for the pin files
    filenames += list(keys_directory + f"/pins/" + str(j).zfill(len(str(pin_num_key_files - 1))) + '_pins.json' for j in range(pin_num_key_files))

    # Generate pin files using multiprocessing for improved performance
    with multiprocessing.Pool(cpu_cores) as pool:
        for _ in tqdm.tqdm(pool.imap(key_generator.keygen_json_pins, filenames), total=pin_num_key_files):
            pass