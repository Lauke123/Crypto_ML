#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import multiprocessing
import os
import tqdm

from _1_keygen_json import KeyGen #  module for key generation

if __name__ == "__main__":

    # Configuration parameters

    keys_per_file = 1000
    num_key_files = 20

    NUMBER_CORS = os.cpu_count() # Number of CPU cores to use for multiprocessing

    # Define directory paths for storing generated data
    current_directory = os.getcwd()
    WORKING_DIR = os.path.join(current_directory, "..")
    PATH_DATA = os.path.join(WORKING_DIR, "Data") 
    PATH_KEYS = os.path.join(PATH_DATA, "1_keys_test")
    PATH_LUGS= os.path.join(PATH_KEYS, "lugs")
    PATH_SORTED_LUGS= os.path.join(PATH_KEYS, "lugs_sorted")

    # Ensure the existence of the directories
    os.makedirs(PATH_KEYS, exist_ok=True)
    os.makedirs(PATH_LUGS, exist_ok=True)
    os.makedirs(PATH_SORTED_LUGS, exist_ok=True)



    # Begin generation of random keys
    print("\n\nGenerating random keys")
    key_generator = KeyGen(count=int(keys_per_file))


    # Change the working directory to where lugs will be stored
    os.chdir(PATH_LUGS)
    # Generate lug settings with varying overlaps
    for i in range(1, 13):
        print(f"Generating lug settings with {i} overlap")
        if (i == 5): # Increase the number of key files significantly for keys with more overlaps
            num_key_files=num_key_files*20
            
        key_generator.path = f"overlaps_{str(i)}/" 
        try:
            os.mkdir(key_generator.path)
        except FileExistsError as err:
            pass # Ignore if the directory already exists
        key_generator.min_overlaps = i
        key_generator.max_overlaps = i
        filenames = list(str(j).zfill(len(str(num_key_files - 1))) + f'_lugs_{str(i)}.json'
                        for j in range(num_key_files))


    # Use multiprocessing to generate keys for the specified filenames
        with multiprocessing.Pool(NUMBER_CORS) as pool:
            for _ in tqdm.tqdm(pool.imap(key_generator.keygen_json_lugs, filenames),total=num_key_files):
                pass


    # Generate lug settings with 1 to max overlap
    os.chdir(PATH_LUGS)
    num_key_files=200
    key_generator = KeyGen(count=int(40000))
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

        with multiprocessing.Pool(NUMBER_CORS) as pool:
            for _ in tqdm.tqdm(pool.imap(key_generator.keygen_json_lugs, filenames), total=num_key_files):
                pass
            
    # Reset configuration parameters for pins generation        
    keys_per_file = 1000
    num_key_files = 20


    from collections import defaultdict, Counter
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pathlib import Path
    import json
    import re

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
    start_path = PATH_LUGS
    working_folder = PATH_SORTED_LUGS
    main(start_path, working_folder)



    # Preparing for pin generation
    os.chdir(PATH_KEYS)
    key_generator = KeyGen(count=int(keys_per_file), path="")
    num_key_files=103
    filenames = []
    # Create directory for pins if it doesn't exist
    try:
        os.mkdir(PATH_KEYS + f"/pins/")
    except FileExistsError as err:
        pass

    # Generate filenames for the pin files
    filenames += list(PATH_KEYS + f"/pins/" + str(j).zfill(len(str(num_key_files - 1))) + '_pins.json' for j in range(num_key_files))
    print (filenames)

    # Generate pin files using multiprocessing for improved performance
    with multiprocessing.Pool(NUMBER_CORS) as pool:
        for _ in tqdm.tqdm(pool.imap(key_generator.keygen_json_pins, filenames), total=num_key_files):
            pass