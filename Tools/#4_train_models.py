# Import necessary libraries for data manipulation, machine learning model building, and memory management
import os
import random
import numpy as np
import pandas as pd
import tqdm
import multiprocessing
from sklearn.model_selection import train_test_split
import gc
import torch
from torch import nn
import placeholder_model
import model

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

# if possible use gpu instead of cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        model_pin = model.Model(input_length=seq_len, num_filters=100, depth=5, num_outputs=1)
        model_pin.to(device)
        while True:

            x, y = load_partial_data(20,100)
            targets = y[:, pin]



            epochs = 10
            batch_size = 10
            correct = 0
            criterion = nn.BCELoss()
            optimizer=torch.optim.Adam(model_pin.parameters(),lr=0.001)
            X_train, X_test, y_train, y_test = train_test_split(x, targets, test_size=0.2, random_state=17)

            # test shape of training data, adding an extra dimension so the channel has a dimension in the tensor. 
            # The conv layer in the model expects a channel dimension with size = 1
            X_train = torch.tensor(X_train, device=device).unsqueeze(1)
            X_test = torch.tensor(X_test, device=device).unsqueeze(1)
            y_train = torch.tensor(y_train, device=device)
            y_test = torch.tensor(y_test, device=device)
            print(X_train.size())
            print(len(X_train))
            print(X_test.size())

            test_loss_pin = 0
            # evaluation of the model
            with torch.no_grad():
                eval_pred = model_pin.forward(X_test)
                eval_pred = torch.flatten(eval_pred)
                test_loss_pin = criterion(eval_pred, y_test)
                eval_pred = torch.round(eval_pred)


                for i in range(len(y_test)):
                    if y_test[i] == eval_pred[i]:
                        correct +=1

            test_accuracy_pin = correct / len(y_test)

            # model training
            for ep in range(epochs):
                print(f"Epoch:{ep}")
                for batch in range(batch_size):
                    batch_starting_point = int((float(batch)/ float(batch_size)) * len(X_train))
                    batch_end_Point = int(((float(batch) + 1.0)/ float(batch_size)) * len(X_train))
                    batch_X_train = X_train[batch_starting_point:batch_end_Point]
                    batch_y_train = y_train[batch_starting_point:batch_end_Point]
                    y_pred = model_pin.forward(batch_X_train)
                    y_pred = torch.flatten(y_pred)
                    loss = criterion(y_pred, batch_y_train)


                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            #test_loss_pin /= batch_size
            print(f"Seq Length {seq_len}, Wheel {wheel}, Pin {pin}: Test Loss: {test_loss_pin}, Test Accuracy: {test_accuracy_pin}")

            # Check if the current pin is the first pin
            if pin == start_pin:
                # If the accuracy is higher than 0.6, set the flag to train all pins
                if test_accuracy_pin > 0.5:
                    train_all_pins = True
                else:
                    break  # Skip training the remaining pins for this wheel
            #retraining for pins vs low accuracy        
            if test_accuracy_pin > 0.70:

                # Record the model's accuracy if it's the first pin or if all pins are being trained
                #if train_all_pins or pin == start_pin:
                model_accuracies.append({
                    'Sequence Length': seq_len,
                    'Wheel Number': wheel,
                    'Pin Number': pin,
                    'Accuracy': test_accuracy_pin
                })

                # Save the final model
                torch.save(model_pin, PATH_MODELS + f'/best_model_wheel_{wheel}_pin_{pin}.pth')

                # Clear the TensorFlow session and collect garbage to free memory
                torch.cuda.empty_cache()
                gc.collect()
                break

            
# Convert the list of accuracies to a DataFrame
accuracy_df = pd.DataFrame(model_accuracies)

# Save the DataFrame to an Excel file
accuracy_df.to_excel('model_accuracies.xlsx', index=False)

# Same code as above but for different sequence length

'''
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
            checkpoint_cb_pin = callbacks.ModelCheckpoint(PATH_MODELS+f'/best_model_wheel_{wheel}_pin_{pin}.keras', save_best_only=True)
            lr_scheduler_cb_pin = callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 0.95 ** epoch)

            # Train the model
            history_pin = model_pin.fit(X_train, y_train, batch_size=260, epochs=1, validation_split=0.2, callbacks=[checkpoint_cb_pin, lr_scheduler_cb_pin])
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
            checkpoint_cb_pin = callbacks.ModelCheckpoint(PATH_MODELS+f'/best_model_wheel_{wheel}_pin_{pin}.keras', save_best_only=True)
            lr_scheduler_cb_pin = callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 0.95 ** epoch)

            # Train the model
            history_pin = model_pin.fit(X_train, y_train, batch_size=520, epochs=1, validation_split=0.2, callbacks=[checkpoint_cb_pin, lr_scheduler_cb_pin])
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

'''