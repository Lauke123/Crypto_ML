# Import necessary libraries for data manipulation, machine learning model building, and memory management
import os
import random
import numpy as np
import pandas as pd
import tqdm
import multiprocessing

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.regularizers import l2
import gc

from tensorflow import keras
from tensorflow.keras import layers
# Constants for defining the range of overlaps as a string
NUMBER_OF_OVERLAPS = "1-12"

# Determine the current working directory
current_directory = os.getcwd()
# Define the working directory as one level up from the current directory
WORKING_DIR = os.path.join(current_directory, "..")

# Path configurations for data storage and model saving
PATH_DATA = os.path.join(WORKING_DIR, "Data") 
PATH_TRAINING_DATA = os.path.join(PATH_DATA, "3_data_npy_train")
PATH_TRAINING_DATA = os.path.join(PATH_TRAINING_DATA, NUMBER_OF_OVERLAPS)
PATH_MODELS_PARENT = os.path.join(PATH_DATA, "models")

# Ensure the model directory exists, create it if not
os.makedirs(PATH_MODELS_PARENT, exist_ok=True)

# Loading Filelist of Training-data
filelist = os.listdir(PATH_TRAINING_DATA)

filelist = [(x,y) for x in filelist if '_x_' in x for y in filelist if x.split('_')[0] == y.split('_')[0] and "_y_" in y]
filelist.sort()

def load_data(count):

    files = random.sample(filelist, k=count)

    x_lst = []
    y_lst = []

    for file in tqdm.tqdm(files):
        x_tmp = np.load(PATH_TRAINING_DATA + "/" + file[0])
        y_tmp = np.load(PATH_TRAINING_DATA+ "/" + file[1])

        if INPUT_SIZE > x_tmp.shape[1]:
            raise UserWarning("Length to height")

        x_tmp = x_tmp[:,:INPUT_SIZE]

        x_lst.append(x_tmp)
        y_lst.append(y_tmp)

    x = np.concatenate(x_lst, axis=0)
    y = np.concatenate(y_lst, axis=0)

    print("calculating float32")
    x = np.subtract(x ,65)
    x = np.array(x, dtype='float32')
    print (x)
    print (y)
    x = np.divide(x , 25)

    y = np.array(y, dtype='float32')

    return x, y

def load_partial_data(count, records_per_file):
    """
    Load a specific number of records from a set number of files.

    Parameters:
    - count: The number of files to randomly select and load data from.
    - records_per_file: The number of records to load from each file.

    Returns:
    - Tuple of np.arrays: The loaded x and y data.
    """

    # Assuming filelist is a list of tuples/lists with paths for x and y data files
    files = random.sample(filelist, k=count)

    x_lst = []
    y_lst = []

    for file in tqdm.tqdm(files):
        # Load only the specified number of records from each file
        x_tmp = np.load(PATH_TRAINING_DATA + "/" + file[0])[:records_per_file]
        y_tmp = np.load(PATH_TRAINING_DATA  + "/" + file[1])[:records_per_file]

        # Ensure the input size matches expected dimensions, adjust if necessary
        if INPUT_SIZE > x_tmp.shape[1]:
            raise UserWarning("Input size too large for loaded data.")

        x_tmp = x_tmp[:, :INPUT_SIZE] # Trim or expand the x data to the INPUT_SIZE


        x_lst.append(x_tmp)
        y_lst.append(y_tmp)
        
    # Concatenate all loaded data into a single array for both x and y
    x = np.concatenate(x_lst, axis=0)
    y = np.concatenate(y_lst, axis=0)

    # Preprocess the data as before
    print("Calculating float32...")
    x = np.subtract(x, 65)
    x = np.array(x, dtype='float32')
    x = np.divide(x, 25)
    
    # Convert y data to float32
    y = np.array(y, dtype='float32')

    return x, y

# Define your ResNet-like architecture
def make_resnet(input_length, num_filters=32, num_outputs=1, d1=512, d2=512, ks=5, depth=5, reg_param=0.0002, final_activation='sigmoid'):
    inp = layers.Input(shape=(input_length, 1))
    
    # First convolutional layer
    conv0 = layers.Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(inp)
    conv0 = layers.BatchNormalization()(conv0)
    conv0 = layers.Activation('relu')(conv0)

    # Residual blocks
    shortcut = conv0
    for i in range(depth):
        conv1 = layers.Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = layers.BatchNormalization()(conv1)
        conv1 = layers.Activation('relu')(conv1)
        conv2 = layers.Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(conv1)
        conv2 = layers.BatchNormalization()(conv2)
        conv2 = layers.Activation('relu')(conv2)
        shortcut = layers.Add()([shortcut, conv2])

    # Output layers
    flat1 = layers.Flatten()(shortcut)
    dense1 = layers.Dense(d1, kernel_regularizer=l2(reg_param))(flat1)
    dense1 = layers.BatchNormalization()(dense1)
    dense1 = layers.Activation('relu')(dense1)
    dense2 = layers.Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    dense2 = layers.Activation('relu')(dense2)
    out = layers.Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2)

    model = models.Model(inputs=inp, outputs=out)
    return model

# Import necessary libraries for data manipulation, machine learning model building, and memory management


# Define input size for the model and the path where models will be saved
INPUT_SIZE = 200
PATH_MODELS =os.path.join(PATH_MODELS_PARENT,f"models_seq_{INPUT_SIZE}")
os.makedirs(PATH_MODELS, exist_ok=True)

#pinwheel_sizes = [26, 25, 23, 21, 19, 17]
pinwheel_sizes = [26]

# Cumulative sizes used for iterating over segments
cumulative_sizes = np.cumsum([0] + pinwheel_sizes)


model_accuracies = []

# Loop over each sequence length
seq_len = INPUT_SIZE  # Define sequence length
# Iterate over each wheel
for wheel, (start_pin, end_pin) in enumerate(zip(cumulative_sizes, cumulative_sizes[1:])):
    train_all_pins = False  # Flag to determine if all pins should be trained

    # Iterate over each pin in the current wheel
    for pin in range(start_pin, end_pin):
        while True:
            # Extract the target values for the current pin
            x, y = load_partial_data(100,15000)

            targets = y[:, pin]
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(x, targets, test_size=0.2, random_state=17)

            # Create and compile the ResNet model
            model_pin = make_resnet(seq_len, num_filters=100, depth=5, num_outputs=1)
            model_pin.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

            # Setup callbacks for model saving and learning rate scheduling
            checkpoint_cb_pin = callbacks.ModelCheckpoint(PATH_MODELS+f'/best_model_wheel_{wheel}_pin_{pin}.h5', save_best_only=True)
            lr_scheduler_cb_pin = callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 0.95 ** epoch)

            # Train the model
            history_pin = model_pin.fit(X_train, y_train, batch_size=1000, epochs=10, validation_split=0.2, callbacks=[checkpoint_cb_pin, lr_scheduler_cb_pin])
            # Evaluate the model on the test set
            test_loss_pin, test_accuracy_pin = model_pin.evaluate(X_test, y_test)
            print(f"Seq Length {seq_len}, Wheel {wheel}, Pin {pin}: Test Loss: {test_loss_pin}, Test Accuracy: {test_accuracy_pin}")

            # Check if the current pin is the first pin
            if pin == start_pin:
                # If the accuracy is higher than 0.6, set the flag to train all pins
                if test_accuracy_pin > 0.5:
                    train_all_pins = True
                else:
                    break  # Skip training the remaining pins for this wheel
            #retraining for pins vs low accuracy        
            if test_accuracy_pin > 0.88:

                # Record the model's accuracy if it's the first pin or if all pins are being trained
                if train_all_pins or pin == start_pin:
                    model_accuracies.append({
                        'Sequence Length': seq_len,
                        'Wheel Number': wheel,
                        'Pin Number': pin,
                        'Accuracy': test_accuracy_pin
                    })

                    # Save the final model
                    model_pin.save(PATH_MODELS+f'/best_model_wheel_{wheel}_pin_{pin}.h5')

                # Clear the TensorFlow session and collect garbage to free memory
                tf.keras.backend.clear_session()
                gc.collect()
                break

            
# Convert the list of accuracies to a DataFrame
accuracy_df = pd.DataFrame(model_accuracies)

# Save the DataFrame to an Excel file
accuracy_df.to_excel('model_accuracies.xlsx', index=False)

# Same code as above but for different sequence length


#Sequence lenght
INPUT_SIZE = 104
PATH_MODELS =os.path.join(PATH_MODELS_PARENT,f"models_seq_{INPUT_SIZE}")
os.makedirs(PATH_MODELS, exist_ok=True)

pinwheel_sizes = [26]

cumulative_sizes = np.cumsum([0] + pinwheel_sizes)


model_accuracies = []

# Loop over each sequence length
seq_len = INPUT_SIZE
# Iterate over each wheel
for wheel, (start_pin, end_pin) in enumerate(zip(cumulative_sizes, cumulative_sizes[1:])):
    train_all_pins = False  # Flag to determine if all pins should be trained

    # Iterate over each pin in the current wheel
    for pin in range(5, end_pin):
        while True:
            # Extract the target values for the current pin
            x, y = load_partial_data(10,1500)

            targets = y[:, pin]
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(x, targets, test_size=0.2, random_state=17)

            # Create and compile the ResNet model
            model_pin = make_resnet(seq_len, num_filters=26, depth=5, num_outputs=1)
            model_pin.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

            # Setup callbacks for model saving and learning rate scheduling
            checkpoint_cb_pin = callbacks.ModelCheckpoint(PATH_MODELS+f'/best_model_wheel_{wheel}_pin_{pin}.h5', save_best_only=True)
            lr_scheduler_cb_pin = callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 0.95 ** epoch)

            # Train the model
            history_pin = model_pin.fit(X_train, y_train, batch_size=260, epochs=10, validation_split=0.2, callbacks=[checkpoint_cb_pin, lr_scheduler_cb_pin])
            # Evaluate the model on the test set
            test_loss_pin, test_accuracy_pin = model_pin.evaluate(X_test, y_test)
            print(f"Seq Length {seq_len}, Wheel {wheel}, Pin {pin}: Test Loss: {test_loss_pin}, Test Accuracy: {test_accuracy_pin}")

            # Check if the current pin is the first pin
            if pin == start_pin:
                # If the accuracy is higher than 0.6, set the flag to train all pins
                if test_accuracy_pin > 0.5:
                    train_all_pins = True
                else:
                    break  # Skip training the remaining pins for this wheel
            #retraining for pins vs low accuracy        
            if test_accuracy_pin > 0.62:

                # Record the model's accuracy if it's the first pin or if all pins are being trained
                if train_all_pins or pin == start_pin:
                    model_accuracies.append({
                        'Sequence Length': seq_len,
                        'Wheel Number': wheel,
                        'Pin Number': pin,
                        'Accuracy': test_accuracy_pin
                    })

                    # Save the final model
                    model_pin.save(PATH_MODELS+f'/best_model_wheel_{wheel}_pin_{pin}.h5')

                # Clear the TensorFlow session and collect garbage to free memory
                tf.keras.backend.clear_session()
                gc.collect()
                break

            
# Convert the list of accuracies to a DataFrame
accuracy_df = pd.DataFrame(model_accuracies)

# Save the DataFrame to an Excel file
accuracy_df.to_excel('model_accuracies.xlsx', index=False)


INPUT_SIZE = 52
PATH_MODELS =os.path.join(PATH_MODELS_PARENT,f"models_seq_{INPUT_SIZE}")
os.makedirs(PATH_MODELS, exist_ok=True)
pinwheel_sizes = [26]

cumulative_sizes = np.cumsum([0] + pinwheel_sizes)


model_accuracies = []

# Loop over each sequence length
seq_len = INPUT_SIZE
# Iterate over each wheel
for wheel, (start_pin, end_pin) in enumerate(zip(cumulative_sizes, cumulative_sizes[1:])):
    train_all_pins = False  # Flag to determine if all pins should be trained

    # Iterate over each pin in the current wheel
    for pin in range(20, end_pin):
        while True:
            # Extract the target values for the current pin
            x, y = load_partial_data(100,10000)

            targets = y[:, pin]
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(x, targets, test_size=0.2, random_state=17)

            # Create and compile the ResNet model
            model_pin = make_resnet(seq_len, num_filters=52, depth=5, num_outputs=1)
            model_pin.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

            # Setup callbacks for model saving and learning rate scheduling
            checkpoint_cb_pin = callbacks.ModelCheckpoint(PATH_MODELS+f'/best_model_wheel_{wheel}_pin_{pin}.h5', save_best_only=True)
            lr_scheduler_cb_pin = callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 0.95 ** epoch)

            # Train the model
            history_pin = model_pin.fit(X_train, y_train, batch_size=520, epochs=10, validation_split=0.2, callbacks=[checkpoint_cb_pin, lr_scheduler_cb_pin])
            # Evaluate the model on the test set
            test_loss_pin, test_accuracy_pin = model_pin.evaluate(X_test, y_test)
            print(f"Seq Length {seq_len}, Wheel {wheel}, Pin {pin}: Test Loss: {test_loss_pin}, Test Accuracy: {test_accuracy_pin}")

            # Check if the current pin is the first pin
            if pin == start_pin:
                # If the accuracy is higher than 0.6, set the flag to train all pins
                if test_accuracy_pin > 0.67:
                    train_all_pins = True
                else:
                    break  # Skip training the remaining pins for this wheel
            #retraining for pins vs low accuracy        
            if test_accuracy_pin > 0.77:

                # Record the model's accuracy if it's the first pin or if all pins are being trained
                if train_all_pins or pin == start_pin:
                    model_accuracies.append({
                        'Sequence Length': seq_len,
                        'Wheel Number': wheel,
                        'Pin Number': pin,
                        'Accuracy': test_accuracy_pin
                    })

                    # Save the final model
                    model_pin.save(PATH_MODELS+f'/best_model_wheel_{wheel}_pin_{pin}.h5')

                # Clear the TensorFlow session and collect garbage to free memory
                tf.keras.backend.clear_session()
                gc.collect()
                break

            
# Convert the list of accuracies to a DataFrame
accuracy_df = pd.DataFrame(model_accuracies)

# Save the DataFrame to an Excel file
accuracy_df.to_excel('model_accuracies.xlsx', index=False)
