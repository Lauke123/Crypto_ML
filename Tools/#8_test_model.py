#!/usr/bin/env python
# coding: utf-8



# Essential library imports for handling data, machine learning models, and plotting
import os
import random
import numpy as np
# import pandas as pd
import tqdm
# import multiprocessing
from matplotlib import pyplot as plt  # Add this line for plotting
import re
import torch

# Define paths to the working directories and data
current_directory = os.getcwd()
WORKING_DIR = os.path.join(current_directory, "..")

PATH_DATA = os.path.join(WORKING_DIR, "Data") 
PATH_TEST_PLOTS = os.path.join(PATH_DATA, "Plots")
PATH_TEST_PLOTS_52 = os.path.join(PATH_TEST_PLOTS, "model_52")
PATH_TEST_PLOTS_104 = os.path.join(PATH_TEST_PLOTS, "model_104")
PATH_TEST_PLOTS_200 = os.path.join(PATH_TEST_PLOTS, "model_200")
os.makedirs(PATH_TEST_PLOTS, exist_ok=True)
os.makedirs(PATH_TEST_PLOTS_52, exist_ok=True)
os.makedirs(PATH_TEST_PLOTS_104, exist_ok=True)
os.makedirs(PATH_TEST_PLOTS_200, exist_ok=True)
PATH_TESTING_DATA = os.path.join(PATH_DATA, "3_data_npy_test")
WHEEL = "Wheel1"
PATH_TESTING_DATA = os.path.join(PATH_TESTING_DATA, WHEEL)
PATH_MODELS_PARENT = os.path.join(PATH_DATA, "models")





INPUT_SIZE = 200 # Fixed input size for the models
PATH_MODELS =os.path.join(PATH_MODELS_PARENT,f"models_seq_{INPUT_SIZE}/")

# if possible use gpu instead of cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load and prepare models for evaluation
model_lst = [file for file in os.listdir(PATH_MODELS) if 'best'  in file]
for i in tqdm.tqdm(range(len(model_lst))):
    myname = model_lst[i]
    model_lst[i] = torch.load(PATH_MODELS+myname)
    model_lst[i].myname = myname
    model_lst[i].to(device)

print(len(model_lst))




# Prepare the testing data file list
filelist = os.listdir(PATH_TESTING_DATA)

# Pairing x and y files
filelist = [
    (x, y) for x in filelist if 'x_' in x
    for y in filelist if x.split('_')[1] == y.split('_')[1]
    and "y_" in y
]

# Function to extract numeric parts from filenames for sorting
def extract_numbers(filename):
    # Extracting numbers after 'overlaps' and 'non-shared-lugs'
    matches = re.findall(r'non-shared-lugs(\d+)-overlaps(\d+)', filename)
    if matches:
        # Converts string numbers to integer tuple
        return tuple(map(int, matches[0]))
    return (0, 0)  # Default return value if no numbers found

# Sorting based on the numeric values extracted from filenames
filelist.sort(key=lambda x: extract_numbers(x[0]))  # sorting by the first file's numbers




print(len(filelist))




def sample_data(x):
    """Randomly sample data to a manageable size."""
    total_samples = len(x)
    sample_size = total_samples
    if total_samples > 1000:
        sample_size = 1000
#    sample_size = min(sample_size, total_samples)
    indices = np.random.choice(total_samples, sample_size, replace=False)
    return x[indices], y[indices]

def compute_predictions(x, model_lst):
    """Compute model predictions for a given dataset."""
    all_predictions = []

    for model in model_lst:
        model_predictions = model.forward(torch.tensor(x, device=device).unsqueeze(1))  # Round predictions to 0 or 1
        model_predictions = torch.round(model_predictions)
        model_predictions = model_predictions.detach().cpu().numpy()
        all_predictions.append(model_predictions.flatten())
    return np.array(all_predictions).T  # Transpose so that each row represents a sample

def count_correct_predictions(all_predictions, y):
    """Count the number of correct predictions per sample."""
    correct_counts = []

    for i in range(len(x)):
        correct_count = 0
        for j, model in enumerate(model_lst):
            if all_predictions[i][j] == y[i][int(model.myname.split('_')[-1].split('.')[0])]:
                correct_count += 1
        correct_counts.append(correct_count)

    return correct_counts




accuracy_results = []

# Evaluation loop for each file pair in the testing dataset
for file in filelist:
    pattern = r'non-shared-lugs(\d+)-overlaps(\d+)'
    match = re.search(pattern, file[1])
    if match:
        non_shared_value,overlaps_value = match.groups()
        print(f"Non-shared lugs: {non_shared_value}. Overlaps: {overlaps_value}")
    
    
       
    x = np.load(PATH_TESTING_DATA + '/' + file[0])
    y = np.load(PATH_TESTING_DATA + '/' + file[1])
    
    num_samples_to_select=1000
    if len(x) < num_samples_to_select:
             num_samples_to_select = (len(x))

    random_indices = random.sample(range(len(x)), num_samples_to_select)
    
    x = x[random_indices]
    y = y[random_indices]  
    
    if INPUT_SIZE > x.shape[1]:
        raise UserWarning("Length too short")

    x = x[:,:INPUT_SIZE]
    
    x = np.subtract(x ,65)
    x = np.array(x, dtype='float32')
    x = np.divide(x , 25)
    y = np.array(y, dtype='float32')
    
    

    all_predictions = compute_predictions(x, model_lst)
    correct_counts =  count_correct_predictions(all_predictions, y)
    
    accuracies = [(count / 26) * 100 for count in correct_counts]
    
    # Calculate mean and median of accuracies
    mean_accuracy = np.mean(accuracies)
    median_accuracy = np.median(accuracies)

    # Display mean and median
    pattern = r'overlaps(\d+)-non-shared-lugs(\d+)'
    # Store the results
    accuracy_results.append((overlaps_value, non_shared_value, mean_accuracy, median_accuracy))
        

    print(f"Mean Accuracy: {mean_accuracy:.2f}%")
    print(f"Median Accuracy: {median_accuracy:.2f}%")
   
    plt.clf()  # Clear the previous figure
    plt.hist(accuracies, bins=53, color='blue',  edgecolor='black')
    plt.title("Distribution of Prediction Accuracies")
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Frequency")
    plt.savefig(PATH_TEST_PLOTS_52 + "/" + f"Non_shared_lugs_{non_shared_value}_Overlaps_{overlaps_value}")
    
    # Second Histogram
    plt.clf()  # Clear the previous figure again before the next plot
    counts, bin_edges, _ = plt.hist(accuracies, bins=range(0, 110, 10), color='blue', alpha=0.7, edgecolor='black')
    total_samples = sum(counts)
    print (f"Number of tested sequences of this type: {total_samples}")  

    # Outputting histogram values as percentages of the total
    for i in range(len(counts)):
        percentage = (counts[i] / total_samples) * 100
        print(f"Accuracy range: {bin_edges[i]}% - {bin_edges[i+1]}%, Frequency: {percentage:.2f}%")  




print("Overlaps | Non-Shared Lugs | Mean Accuracy | Median Accuracy")
print("-" * 50)
for result in accuracy_results:
    print(f"{result[0]} | {result[1]} | {result[2]:.2f}% | {result[3]:.2f}%")




def load_data(count):

    files = random.sample(filelist, k=count)

    x_lst = []
    y_lst = []

    for file in tqdm.tqdm(files):
        
        x_tmp = np.load(PATH_TESTING_DATA + '/' + file[0])
        y_tmp = np.load(PATH_TESTING_DATA + '/' + file[1])

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
    x = np.divide(x , 25)

    y = np.array(y, dtype='float32')

    return x, y




import numpy as np
import matplotlib.pyplot as plt
import tqdm
# Assume load_data and model_lst are defined elsewhere in your code

def sample_data(X, y, sample_rate=0.01):
    total_samples = len(X)
    sample_size = int(total_samples * sample_rate)
    indices = np.random.choice(total_samples, sample_size, replace=False)
    return X[indices], y[indices]

def compute_predictions(x, model_lst):
    """Compute model predictions for a given dataset."""
    all_predictions = []

    for model in model_lst:
        model_predictions = model.forward(torch.tensor(x, device=device).unsqueeze(1))  # Round predictions to 0 or 1
        model_predictions = torch.round(model_predictions)
        model_predictions = model_predictions.detach().cpu().numpy()
        all_predictions.append(model_predictions.flatten())
    return np.array(all_predictions).T  # Transpose so that each row represents a sample


def count_correct_predictions(all_predictions, y, X, model_lst):
    correct_counts = []
    for i in range(len(X)):
        correct_count = 0
        for j, model in enumerate(model_lst):
            if all_predictions[i][j] == y[i][int(model.myname.split('_')[-1].split('.')[0])]:
                correct_count += 1
        correct_counts.append(correct_count)
    return correct_counts

# Assume X, y, and model_lst are properly initialized
#X, y = sample_data(X, y)
X, y = load_data(103)
all_predictions = compute_predictions(X, model_lst)
correct_counts = count_correct_predictions(all_predictions, y, X, model_lst)

# Compute accuracies
accuracies = [(count / len(model_lst)) * 100 for count in correct_counts]

# Calculate average accuracy
average_accuracy = np.mean(accuracies)

# Plotting the accuracy distribution
plt.clf()  # Clear the previous figure
counts, bin_edges, _ = plt.hist(accuracies, bins=53, color='lightgray', alpha=0.7, edgecolor='black')
plt.title("Accuracy distribution")
plt.xlabel("Accuracy (%)")
plt.ylabel("Frequency")

plt.clf()  # Clear the previous figure
# Show average accuracy on the graph
plt.axvline(average_accuracy, color='red', linestyle='dashed', linewidth=1)
plt.text(average_accuracy, max(counts)/2, f'Average: {average_accuracy:.2f}%', rotation=90)
plt.savefig(PATH_TEST_PLOTS_52 + "/" + "average_accuracy")



plt.clf()  # Clear the previous figure
# Adjust the number of bins manually based on your data's characteristics
adjusted_bins = 22  # Example: set to 20, adjust based on your data's distribution
plt.hist(accuracies, bins=adjusted_bins, color='lightgray', alpha=0.7, edgecolor='black')
plt.title("Accuracy distribution for n=52")
plt.xlabel("Accuracy (%)")
plt.ylabel("Frequency")
plt.axvline(average_accuracy, color='red', linestyle='dashed', linewidth=1)
plt.text(average_accuracy, plt.ylim()[1]/2, f'Average: {average_accuracy:.2f}%', rotation=90)
plt.savefig(PATH_TEST_PLOTS_52 + "/" + "average_accuracy_adjusted")



'''
INPUT_SIZE = 104
PATH_MODELS =os.path.join(PATH_MODELS_PARENT,f"models_seq_{INPUT_SIZE}/")




model_lst = [file for file in os.listdir(PATH_MODELS) if 'best'  in file]
for i in tqdm.tqdm(range(len(model_lst))):
    myname = model_lst[i]
    model_lst[i] = keras.models.load_model(PATH_MODELS+myname)
    model_lst[i].myname = myname

print(len(model_lst))




import numpy as np
import matplotlib.pyplot as plt
import tqdm
# Assume load_data and model_lst are defined elsewhere in your code

def sample_data(X, y, sample_rate=0.01):
    total_samples = len(X)
    sample_size = int(total_samples * sample_rate)
    indices = np.random.choice(total_samples, sample_size, replace=False)
    return X[indices], y[indices]

def compute_predictions(X, model_lst):
    all_predictions = []
    for model in tqdm.tqdm(model_lst):
        model_predictions = model.predict(X).round()  # Round predictions to 0 or 1
        all_predictions.append(model_predictions.flatten())
    return np.array(all_predictions).T  # Transpose so that each row represents a sample

def count_correct_predictions(all_predictions, y, X, model_lst):
    correct_counts = []
    for i in range(len(X)):
        correct_count = 0
        for j, model in enumerate(model_lst):
            if all_predictions[i][j] == y[i][int(model.myname.split('_')[-1].split('.')[0])]:
                correct_count += 1
        correct_counts.append(correct_count)
    return correct_counts

# Assume X, y, and model_lst are properly initialized
#X, y = sample_data(X, y)
X, y = load_data(103)
all_predictions = compute_predictions(X, model_lst)
correct_counts = count_correct_predictions(all_predictions, y, X, model_lst)

# Compute accuracies
accuracies = [(count / len(model_lst)) * 100 for count in correct_counts]

# Calculate average accuracy
average_accuracy = np.mean(accuracies)

plt.clf()  # Clear the previous figure
# Plotting the accuracy distribution
counts, bin_edges, _ = plt.hist(accuracies, bins=53, color='lightgray', alpha=0.7, edgecolor='black')
plt.title("Accuracy distribution")
plt.xlabel("Accuracy (%)")
plt.ylabel("Frequency")

# Show average accuracy on the graph
plt.axvline(average_accuracy, color='red', linestyle='dashed', linewidth=1)
plt.text(average_accuracy, max(counts)/2, f'Average: {average_accuracy:.2f}%', rotation=90)

plt.savefig(PATH_TEST_PLOTS_104 + "/" + "average_accuracy")

plt.clf()  # Clear the previous figure
# Adjust the number of bins manually based on your data's characteristics
adjusted_bins = 21  # Example: set to 20, adjust based on your data's distribution
plt.hist(accuracies, bins=adjusted_bins, color='lightgray', alpha=0.7, edgecolor='black')
plt.title("Accuracy distribution for n=104")
plt.xlabel("Accuracy (%)")
plt.ylabel("Frequency")
plt.axvline(average_accuracy, color='red', linestyle='dashed', linewidth=1)
plt.text(average_accuracy, plt.ylim()[1]/2, f'Average: {average_accuracy:.2f}%', rotation=90)
plt.savefig(PATH_TEST_PLOTS_104 + "/" + "average_accuracy_adjusted")




INPUT_SIZE = 200
PATH_MODELS =os.path.join(PATH_MODELS_PARENT,f"models_seq_{INPUT_SIZE}/")




model_lst = [file for file in os.listdir(PATH_MODELS) if 'best'  in file]
for i in tqdm.tqdm(range(len(model_lst))):
    myname = model_lst[i]
    model_lst[i] = keras.models.load_model(PATH_MODELS+myname)
    model_lst[i].myname = myname

print(len(model_lst))




import numpy as np
import matplotlib.pyplot as plt
import tqdm
# Assume load_data and model_lst are defined elsewhere in your code

def sample_data(X, y, sample_rate=0.01):
    total_samples = len(X)
    sample_size = int(total_samples * sample_rate)
    indices = np.random.choice(total_samples, sample_size, replace=False)
    return X[indices], y[indices]

def compute_predictions(X, model_lst):
    all_predictions = []
    for model in tqdm.tqdm(model_lst):
        model_predictions = model.predict(X).round()  # Round predictions to 0 or 1
        all_predictions.append(model_predictions.flatten())
    return np.array(all_predictions).T  # Transpose so that each row represents a sample

def count_correct_predictions(all_predictions, y, X, model_lst):
    correct_counts = []
    for i in range(len(X)):
        correct_count = 0
        for j, model in enumerate(model_lst):
            if all_predictions[i][j] == y[i][int(model.myname.split('_')[-1].split('.')[0])]:
                correct_count += 1
        correct_counts.append(correct_count)
    return correct_counts

# Assume X, y, and model_lst are properly initialized
#X, y = sample_data(X, y)
X, y = load_data(103)
all_predictions = compute_predictions(X, model_lst)
correct_counts = count_correct_predictions(all_predictions, y, X, model_lst)

# Compute accuracies
accuracies = [(count / len(model_lst)) * 100 for count in correct_counts]

# Calculate average accuracy
average_accuracy = np.mean(accuracies)

plt.clf()  # Clear the previous figure
# Plotting the accuracy distribution
counts, bin_edges, _ = plt.hist(accuracies, bins=53, color='lightgray', alpha=0.7, edgecolor='black')
plt.title("Accuracy distribution")
plt.xlabel("Accuracy (%)")
plt.ylabel("Frequency")

# Show average accuracy on the graph
plt.axvline(average_accuracy, color='red', linestyle='dashed', linewidth=1)
plt.text(average_accuracy, max(counts)/2, f'Average: {average_accuracy:.2f}%', rotation=90)

plt.savefig(PATH_TEST_PLOTS_200 + "/" + "average_accuracy")



plt.clf()  # Clear the previous figure
# Adjust the number of bins manually based on your data's characteristics
adjusted_bins = 20  # Example: set to 20, adjust based on your data's distribution
plt.hist(accuracies, bins=adjusted_bins, color='lightgray', alpha=0.7, edgecolor='black')
plt.title("Accuracy distribution for n=200")
plt.xlabel("Accuracy (%)")
plt.ylabel("Frequency")
plt.axvline(average_accuracy, color='red', linestyle='dashed', linewidth=1)
plt.text(average_accuracy, plt.ylim()[1]/2, f'Average: {average_accuracy:.2f}%', rotation=90)
plt.savefig(PATH_TEST_PLOTS_200 + "/" + "average_accuracy_adjusted")




accuracies = [(count / 26) * 100 for count in correct_counts]
'''