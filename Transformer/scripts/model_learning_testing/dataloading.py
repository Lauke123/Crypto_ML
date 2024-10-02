import numpy as np
import random
import tqdm


def load_partial_data(count,filelist, path_data, inputsize, records_per_file=None, lugs=False):
    """Load and return part of the training and testing data.

    Parameters
    ----------
    count:
        The number of files to randomly select and load data from.
    path_data: str
        path to the folder the files in filelist are from.
    inputsize: int
        inputsize of the model, necesaary to trim data to the right shape.
    records_per_file:
        The number of records to load from each file.
    lugs: bool
        if true append the amount of all different possible lug pairs and the targets

    Returns
    -------
    - Tuple of np.arrays: The loaded x and y data.

    """
    # Assuming filelist is a list of tuples/lists with paths for x and y data files
    files = random.sample(filelist, k=count)

    x_lst = []
    y_lst = []

    for file in tqdm.tqdm(files):
        # Load only the specified number of records from each file
        if records_per_file == None:
            x_tmp = np.load(path_data + '/' + file[0])
            y_tmp = np.load(path_data + '/' + file[1])
            y_lugs_tmp = np.load(path_data + '/' + file[2])
        else:
            x_tmp = np.load(path_data + "/" + file[0])[:records_per_file]
            y_tmp = np.load(path_data  + "/" + file[1])[:records_per_file]
            y_lugs_tmp = np.load(path_data + '/' + file[2])[:records_per_file]
        # Ensure the input size matches expected dimensions, adjust if necessary
        if inputsize > x_tmp.shape[1]:
            raise UserWarning("Input size too large for loaded data.")

        x_tmp = x_tmp[:, :inputsize] # Trim or expand the x data to the INPUT_SIZE


        x_lst.append(x_tmp)
        # add lug position values to the target for training to predict lug settings in addition to pin settings
        if lugs:
            y_tmp = np.concatenate((y_tmp, y_lugs_tmp), axis=1)
        y_lst.append(y_tmp)

    # Concatenate all loaded data into a single array for both x and y
    x = np.concatenate(x_lst, axis=0)
    y = np.concatenate(y_lst, axis=0)

    # Preprocess the data as before
    print("Calculating float32...")
    x = np.subtract(x, 65)
    x = np.array(x, dtype='int32')


    # Convert y data to float32
    y = np.array(y, dtype='float32')

    return x, y
