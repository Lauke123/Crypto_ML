# Import necessary libraries for data manipulation, machine learning model building, and memory management
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import gc
import torch
from torch import nn
import model
from dataloading import load_partial_data
from model_learning import Learner
from progress_plot import ProgressPlot

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

# if possible use gpu instead of cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Taining is using GPU instead of CPU")

# Define input size for the model and the path where models will be saved
INPUT_SIZE = 200
epochs = 10
batch_size = 1000
PATH_MODELS =os.path.join(PATH_MODELS_PARENT,f"models_seq_{INPUT_SIZE}")
os.makedirs(PATH_MODELS, exist_ok=True)

#pinwheel_sizes = [26, 25, 23, 21, 19, 17]
pinwheel_sizes = [26]

# Cumulative sizes used for iterating over segments
cumulative_sizes = np.cumsum([0] + pinwheel_sizes)

progressplot = ProgressPlot()

model_accuracies = []

# Loop over each sequence length
seq_len = INPUT_SIZE  # Define sequence length
# Iterate over each wheel
for wheel, (start_pin, end_pin) in enumerate(zip(cumulative_sizes, cumulative_sizes[1:])):
    train_all_pins = False  # Flag to determine if all pins should be trained

    # Iterate over each pin in the current wheel

    for pin in range(start_pin, end_pin):
        model_pin = model.Model(input_length=seq_len, num_filters=100, depth=5, num_outputs=1)
        learner = Learner(model_pin)
        model_pin.to(device)
        progress_data = []
        while True:

            x, y = load_partial_data(10,15000, filelist, PATH_TRAINING_DATA, INPUT_SIZE)
            targets = y[:, pin]
            print("Data is loaded")

            X_train, X_test, y_train, y_test = train_test_split(x, targets, test_size=0.2, random_state=17)

            # test shape of training data, adding an extra dimension so the channel has a dimension in the tensor. 
            # The conv layer in the model expects a channel dimension with size = 1
            X_train = torch.tensor(X_train, device=device).unsqueeze(1)
            X_test = torch.tensor(X_test, device=device).unsqueeze(1)
            y_train = torch.tensor(y_train, device=device)
            y_test = torch.tensor(y_test, device=device)



            # model training
            learner.fit(batch_size, y_train, X_train, epochs, True)

            # model evaluation
            test_loss_pin, test_accuracy_pin = learner.evaluate(X_test, y_test)
            progress_data.append(test_accuracy_pin)

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
            if test_accuracy_pin > 0.88:

                # Record the model's accuracy if it's the first pin or if all pins are being trained
                #if train_all_pins or pin == start_pin:
                model_accuracies.append({
                    'Sequence Length': seq_len,
                    'Wheel Number': wheel,
                    'Pin Number': pin,
                    'Accuracy': test_accuracy_pin
                })

                progressplot.append_data(progress_data)

                # Save the final model
                torch.save(model_pin, PATH_MODELS + f'/best_model_wheel_{wheel}_pin_{pin}.pth')

                # Clear the TensorFlow session and collect garbage to free memory
                torch.cuda.empty_cache()
                gc.collect()
                break

progressplot.generate_plot()
            
# Convert the list of accuracies to a DataFrame
accuracy_df = pd.DataFrame(model_accuracies)

# Save the DataFrame to an Excel file
accuracy_df.to_excel('model_accuracies.xlsx', index=False)
