import os, shutil
import random
import multiprocessing

from _3_encrypt import Encrypt

# Initialize the Encrypt class with specified parameters
m209 = Encrypt(destin_path="", count_a=500,
            append_ciphertext=False,
            append_plaintext=False,
            append_keystream=True,
            append_lugs=True,
            append_pins=True)

# Define a function to load keys and perform encryption using multiprocessing
# defined before the guard to enable access for child proceses to that function
def generate_data(x):
    print(f"working with {x}")
    filename, i, files_count = x
    m209.load_keys(filename, filetype="pins_n_lugs")
    m209.set_int_msg_ind("AAAAAA")
    m209.encrypt(f"{str(i).zfill(len(str(files_count-1)))}")

# guard to avoid child processes executing the entire code 
# ( important if executed on windows because default is spwan and not fork)
def create_ciphertext_train(cpu_cores: int, output_path: str, number_of_files: int = 20, number_of_overlaps: str = "1-12") -> None:
    """Create the ciphertext in training according to keys stored in the previous step.

    Parameters
    ----------
    cpu_cores : int
        amount of cpu cores that should be used during multiprocessing
    output_path : str
        absolute path to the directory the data folder should be created in.
    number_of_files : int
        the amount of files that are produced, used to replicate the lug_settings x times
    number_of_overlaps: str
        Specifies the range of overlaps to be used in encryption

    Returns
    -------
    None

    """
    output_directory = output_path
    data_directory = os.path.join(output_directory, "Data") 
    keys_directory = os.path.join(data_directory, "1_keys_train")
    ciphertext_directory = os.path.join(data_directory, "2_ciphertexts_train")  # Ensure  directory with ciphertexts used for training exists

    os.makedirs(ciphertext_directory, exist_ok=True)

    # Create or ensure the existence of a specific directory for ciphertexts based on the number of overlaps
    ciphertext_directory = os.path.join(ciphertext_directory, number_of_overlaps)

    os.makedirs(ciphertext_directory, exist_ok=True)


    # Filter and list lug and pin files from the keys directory based on specified criteria
    lug_files = [f for f in os.listdir(keys_directory) if (number_of_overlaps +".") in f] 
    lug_files = [f for f in lug_files if ".tar.gz" in f]
    pin_files = [f for f in os.listdir(keys_directory) if "pins" in f]
    pin_files = [f for f in pin_files if ".tar.gz" in f]

    # Copy lug and pin files to the ciphertexts directory
    os.chdir(ciphertext_directory)
    for file in lug_files:
        shutil.copy(keys_directory + '/' + file, ciphertext_directory)

    for file in pin_files:
        shutil.copy(keys_directory + '/' + file, ciphertext_directory)


    # Extract the contents of the copied .tar.gz files and then remove the archives

    os.chdir(ciphertext_directory+'/')

    for file in os.listdir():
        os.system(f"tar -xvzf {file}")
        os.remove(file)

    # Prepare file paths for lug settings and pin settings by listing them and replicating based on number_of_files

    os.chdir(ciphertext_directory)

    lug_setting_files = [f"overlaps_{number_of_overlaps}/" + s for s in os.listdir(f"overlaps_{number_of_overlaps}/")]
    lug_setting_files = lug_setting_files * number_of_files

    pin_folders = [folder for folder in os.listdir() if "pin" in folder]

    pin_setting_files = []
    for folder in pin_folders:
        pin_setting_files += [folder +'/'+ files for files in os.listdir(folder)]

    os.chdir(ciphertext_directory+"/")
    # Shuffle the lists of lug and pin setting files to randomize the encryption process
    random.shuffle(lug_setting_files)
    random.shuffle(pin_setting_files)

    # Pair each lug setting file with a pin setting file
    files = list(zip(lug_setting_files, pin_setting_files))

    # Use multiprocessing to encrypt data using the paired lug and pin setting files
    with multiprocessing.Pool(cpu_cores) as pool:
        for _ in pool.imap(generate_data, [(files[i], i, len(files)) for i in range(len(files))]):
            pass

    # Clean up the directory by removing extracted lug and pin folders
    os.chdir(ciphertext_directory)
    for x in [f"{i}" for i in os.listdir() if "pins" in i]:
        # os.system(x)
        shutil.rmtree(x)

    for x in [f"{i}" for i in os.listdir() if "overlaps" in i]:
        # os.system(x)
        shutil.rmtree(x)

    # Rename the generated files to include the number of overlaps in their name
    os.chdir(ciphertext_directory)
    for x in [(f"{i}", f"{i.split('.')[0]+'_'+number_of_overlaps}.json") for i in os.listdir()]:
        # os.system(x)
        shutil.move(x[0], x[1])