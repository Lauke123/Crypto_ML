import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
import tqdm
from model_learning_testing.dataloading import load_partial_data
from model_learning_testing.model_testing import ModelTester
from model_learning_testing.plots import PlotGenerator


# Function to extract numeric parts from filenames for sorting
def extract_numbers(filename):
    # Extracting numbers after 'overlaps' and 'non-shared-lugs'
    matches = re.findall(r'non-shared-lugs(\d+)-overlaps(\d+)', filename)
    if matches:
        # Converts string numbers to integer tuple
        return tuple(map(int, matches[0]))
    return (0, 0)  # Default return value if no numbers found

def test_mixed_lugs(model_input_size: int, test_results_directory: str, npy_data_directory: str,
                    model_lst: list, plotgenerator: PlotGenerator, modeltester: ModelTester, filelist: list):
    """Test model on mixed amount of lug overlaps and non-shared lugs."""
    x, y = load_partial_data(103, filelist=filelist,
                            path_data=npy_data_directory, inputsize=model_input_size)

    all_predictions_mixed_overlaps = modeltester.compute_predictions(x, 1000)
    correct_counts_mixed_overlaps = modeltester.count_correct_predictions(all_predictions_mixed_overlaps, y, x)

    # Compute accuracies of the dataset that has mixed overlaps
    accuracies_mixed_overlaps = [(count / len(model_lst)) * 100 for count in correct_counts_mixed_overlaps]

    # save accuracies to csv file
    accuracy_df = pd.DataFrame(accuracies_mixed_overlaps)
    accuracy_df.to_csv(f"{test_results_directory}/prediction_accuracies_{model_input_size}.csv", index=False)

    # Calculate mean and median of accuracies
    mean_accuracy_mixed_overlaps = np.mean(accuracies_mixed_overlaps)
    median_accuracy_mixed_overlaps = np.median(accuracies_mixed_overlaps)

    # save the results of the accuracies vor varying overlaps in csv file
    csv_file = os.path.join(test_results_directory, "accuracies_mixed_overlaps.csv")
    accuracy_mixed_df = pd.DataFrame([{
        "mean_accuracy_mixed_overlaps" : mean_accuracy_mixed_overlaps,
        "median_accuracy_mixed_overlaps": median_accuracy_mixed_overlaps}])
    accuracy_mixed_df.to_csv(csv_file, index=False)

    plotgenerator.generate_plot(accuracies_mixed_overlaps,
                                f"Accuracy distribution for n={model_input_size}",
                                "average_accuracies_mixed_overlaps")
    plotgenerator.generate_plot(accuracies_mixed_overlaps,
                                f"Accuracy distribution for n={model_input_size}",
                                "average_accuracies_mixed_overlaps_adjusted_bins",bins=20)

def test_varying_lugs(test_results_directory: str, npy_data_directory: str, plotgenerator: PlotGenerator,
                      modeltester: ModelTester, filelist: list):
    """Test model on different varying of lug overlaps and non-shared lugs."""
    accuracy_results = []
    # Evaluation loop for each file pair in the testing dataset
    for file in filelist:
        pattern = r'non-shared-lugs(\d+)-overlaps(\d+)'
        match = re.search(pattern, file[1])
        if match:
            non_shared_value,overlaps_value = match.groups()

        x = np.load(npy_data_directory + '/' + file[0])
        y = np.load(npy_data_directory + '/' + file[1])

        num_samples_to_select=1000
        if len(x) < num_samples_to_select:
                num_samples_to_select = (len(x))

        x, y = modeltester.sample_data(x, y, num_samples_to_select)

        x, y = modeltester.normalize_data(x, y)

        all_predictions = modeltester.compute_predictions(x, len(x))
        correct_counts =  modeltester.count_correct_predictions(all_predictions, y, x)

        accuracies = [(count / 26) * 100 for count in correct_counts]

        # Calculate mean and median of accuracies
        mean_accuracy = np.mean(accuracies)
        median_accuracy = np.median(accuracies)

        # Store the results to later save it in csv file with panda Dataframe
        accuracy_results.append(
            {
            "overlaps_value": overlaps_value,
            "non_shared_value" : non_shared_value,
            "mean_accuracy" : mean_accuracy,
            "median_accuracy" : median_accuracy})

        plotgenerator.generate_plot(accuracies,
                                    f"Non_shared_lugs_{non_shared_value}_Overlaps_{overlaps_value}",
                                    f"Non_shared_lugs_{non_shared_value}_Overlaps_{overlaps_value}",
                                    )

    # save the results of the accuracies vor varying overlaps in csv file
    accuracy_df = pd.DataFrame(accuracy_results)
    csv_file = os.path.join(test_results_directory, "accuracies_varying_overlaps.csv")
    accuracy_df.to_csv(csv_file, index=False)

def testing(output_directory_path: str, model_input_size: int = 200) -> None:
    """Test the models in the given folder with the testing data in that folder.

    Parameters
    ----------
    output_directory_path: str
        the directory the data folder is in that was created during the execution of create_dataset.py
    model_input_size : int
        the size of the input that model was trained with

    Returns
    -------
    None

    """
    # Define paths to the working directories and data
    output_directory = output_directory_path
    data_directory = os.path.join(output_directory, "Data")
    test_results_directory = os.path.join(output_directory, f"test_results/model_{model_input_size}")
    plots_input_size_directory = os.path.join(test_results_directory, "Plots")
    npy_data_directory = os.path.join(data_directory, "3_data_npy_test")
    wheel = "Wheel1"
    npy_data_directory = os.path.join(npy_data_directory, wheel)
    model_directory = os.path.join(data_directory, "models")
    model_input_size_directory =os.path.join(model_directory,f"models_seq_{model_input_size}/")
    # create new directory for the plots generated during testing 
    os.makedirs(plots_input_size_directory, exist_ok=True)

    # if possible use gpu instead of cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load and prepare models for evaluation
    model_lst = [file for file in os.listdir(model_input_size_directory) if 'best'  in file]
    for i in tqdm.tqdm(range(len(model_lst))):
        myname = model_lst[i]
        model_lst[i] = torch.load(model_input_size_directory+myname)
        model_lst[i].myname = myname
        model_lst[i].to(device)

    plotgenerator = PlotGenerator(plots_input_size_directory)
    modeltester = ModelTester(model_lst, device, model_input_size)

    # Prepare the testing data file list
    filelist = os.listdir(npy_data_directory)
    # Pairing x and y files
    filelist = [(x,y,z) for x in filelist if 'x_' in x for y in filelist if x.split('_5')[1] == y.split('_5')[1] and "y_5" in y for z in filelist if y.split('_5')[1] == z.split('_5')[1] and "y_lugs" in z]
    # Sorting based on the numeric values extracted from filenames
    filelist.sort(key=lambda x: extract_numbers(x[0]))  # sorting by the first file's numbers

    # test the model's accuracies for a specific amount of overlaps and non-shared lugs
    test_varying_lugs(test_results_directory, npy_data_directory, plotgenerator,
                      modeltester, filelist)
    # test the model's accuracies for a random mix of all possible overlaps and non-shared lugs
    test_mixed_lugs(model_input_size, test_results_directory, npy_data_directory,
                    model_lst, plotgenerator, modeltester, filelist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_folder_path", type=str, help="path to the folder the data folder was created in during the create_dataset.py")
    parser.add_argument("-m", "--model_size", type=int, default=200, help="size of the input layer of the model that is being tested" )
    args = parser.parse_args()
    testing(output_directory_path=args.output_folder_path, model_input_size=args.model_size)
