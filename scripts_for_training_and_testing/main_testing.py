import os
import random
import numpy as np
import tqdm
import re
import torch
from plots import PlotGenerator
from model_testing import ModelTester
from dataloading import load_partial_data

# Define paths to the working directories and data
current_directory = os.getcwd()
WORKING_DIR = os.path.join(current_directory, "..")

PATH_DATA = os.path.join(WORKING_DIR, "Data") 
PATH_TEST_PLOTS = os.path.join(PATH_DATA, "Plots")
PATH_TEST_PLOTS = os.path.join(PATH_TEST_PLOTS, "model_200")
os.makedirs(PATH_TEST_PLOTS, exist_ok=True)
os.makedirs(PATH_TEST_PLOTS, exist_ok=True)
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

plotgenerator = PlotGenerator(PATH_TEST_PLOTS)
modeltester = ModelTester(model_lst, device)

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


    all_predictions = modeltester.compute_predictions(x)
    correct_counts =  modeltester.count_correct_predictions(all_predictions, y, x)

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

    plotgenerator.generate_plot(accuracies,
                                f"Non_shared_lugs_{non_shared_value}_Overlaps_{overlaps_value}",
                                f"Non_shared_lugs_{non_shared_value}_Overlaps_{overlaps_value}",
                                )

    '''
    counts, bin_edges, _ = plt.hist(accuracies, bins=range(0, 110, 10), color='blue', alpha=0.7, edgecolor='black')
    total_samples = sum(counts)
    print (f"Number of tested sequences of this type: {total_samples}")  
    # Outputting histogram values as percentages of the total
    for i in range(len(counts)):
        percentage = (counts[i] / total_samples) * 100
        print(f"Accuracy range: {bin_edges[i]}% - {bin_edges[i+1]}%, Frequency: {percentage:.2f}%")  '''




print("Overlaps | Non-Shared Lugs | Mean Accuracy | Median Accuracy")
print("-" * 50)
for result in accuracy_results:
    print(f"{result[0]} | {result[1]} | {result[2]:.2f}% | {result[3]:.2f}%")


# Assume X, y, and model_lst are properly initialized
#X, y = sample_data(X, y)
X, y = load_partial_data(103, filelist=filelist,
                         path_data=PATH_TESTING_DATA, inputsize=INPUT_SIZE)
all_predictions = modeltester.compute_predictions(X)
correct_counts = modeltester.count_correct_predictions(all_predictions, y, X)

# Compute accuracies
accuracies = [(count / len(model_lst)) * 100 for count in correct_counts]

plotgenerator.generate_plot(accuracies,
                            f"Accuracy distribution for n={INPUT_SIZE}",
                            "average_accuracy_adjusted")