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
if __name__ == "__main__":

    # Configuration parameters

    NUMBER_OF_FILES = 20
    NUMBER_OF_OVERLAPS = "1-12" # Specifies the range of overlaps to be used in encryption


    current_directory = os.getcwd()
    WORKING_DIR = os.path.join(current_directory, "..")
    PATH_DATA = os.path.join(WORKING_DIR, "Data") 
    PATH_KEYS = os.path.join(PATH_DATA, "1_keys_train")
    PATH_CIPHERTEXTS = os.path.join(PATH_DATA, "2_ciphertexts_train")  # Ensure  directory with ciphertexts used for training exists

    os.makedirs(PATH_CIPHERTEXTS, exist_ok=True)

    # Create or ensure the existence of a specific directory for ciphertexts based on the number of overlaps
    PATH_CIPHERTEXTS = os.path.join(PATH_CIPHERTEXTS, NUMBER_OF_OVERLAPS)

    os.makedirs(PATH_CIPHERTEXTS, exist_ok=True)

    # Set the number of CPU cores for multiprocessing
    #NUMBER_CORS = multiprocessing.cpu_count()
    NUMBER_CORS = os.cpu_count()
    INPUT_SIZE = 500 # Set the sequence lenghtes

    # Filter and list lug and pin files from the keys directory based on specified criteria

    lug_files = [f for f in os.listdir(PATH_KEYS) if (NUMBER_OF_OVERLAPS +".") in f] 
    lug_files = [f for f in lug_files if ".tar.gz" in f]
    pin_files = [f for f in os.listdir(PATH_KEYS) if "pins" in f]
    pin_files = [f for f in pin_files if ".tar.gz" in f]
    print(lug_files)
    print(pin_files)

    # Copy lug and pin files to the ciphertexts directory
    os.chdir(PATH_CIPHERTEXTS)
    for file in lug_files:
        print (file)
        # os.system(f"cp {PATH_KEYS +'/' +file} {PATH_CIPHERTEXTS+'/'}")
        shutil.copy(PATH_KEYS + '/' + file, PATH_CIPHERTEXTS)

    for file in pin_files:
        print (file)
        # os.system(f"cp {PATH_KEYS+'/'+file} {PATH_CIPHERTEXTS+'/'}")
        shutil.copy(PATH_KEYS + '/' + file, PATH_CIPHERTEXTS)


    # Extract the contents of the copied .tar.gz files and then remove the archives

    os.chdir(PATH_CIPHERTEXTS+'/')

    for file in os.listdir():
        print(file)
        os.system(f"tar -xvzf {file}")
        # os.system(f"rm {file}")
        os.remove(file)


    # Prepare file paths for lug settings and pin settings by listing them and replicating based on NUMBER_OF_FILES

    os.chdir(PATH_CIPHERTEXTS)

    lug_setting_files = [f"overlaps_{NUMBER_OF_OVERLAPS}/" + s for s in os.listdir(f"overlaps_{NUMBER_OF_OVERLAPS}/")]
    lug_setting_files = lug_setting_files * NUMBER_OF_FILES

    pin_folders = [folder for folder in os.listdir() if "pin" in folder]

    pin_setting_files = []
    for folder in pin_folders:
        pin_setting_files += [folder +'/'+ files for files in os.listdir(folder)]


    print (pin_setting_files)

    print (lug_setting_files)


    os.chdir(PATH_CIPHERTEXTS+"/")
    # Shuffle the lists of lug and pin setting files to randomize the encryption process
    random.shuffle(lug_setting_files)
    random.shuffle(pin_setting_files)

    # Pair each lug setting file with a pin setting file
    files = list(zip(lug_setting_files, pin_setting_files))
    print (files)

    # Use multiprocessing to encrypt data using the paired lug and pin setting files
    with multiprocessing.Pool(NUMBER_CORS) as pool:
        for _ in pool.imap(generate_data, [(files[i], i, len(files)) for i in range(len(files))]):
            pass

    # Clean up the directory by removing extracted lug and pin folders
    os.chdir(PATH_CIPHERTEXTS)
    for x in [f"{i}" for i in os.listdir() if "pins" in i]:
        # os.system(x)
        shutil.rmtree(x)

    for x in [f"{i}" for i in os.listdir() if "overlaps" in i]:
        # os.system(x)
        shutil.rmtree(x)

    # Rename the generated files to include the number of overlaps in their name
    os.chdir(PATH_CIPHERTEXTS)
    for x in [(f"{i}", f"{i.split('.')[0]+'_'+NUMBER_OF_OVERLAPS}.json") for i in os.listdir()]:
        # os.system(x)
        shutil.move(x[0], x[1])