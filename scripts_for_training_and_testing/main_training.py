import argparse
import gc
import os

import numpy as np
import pandas as pd
import torch
from model import Model
from model_learning import Learner, LearnerDataset


def training(output_directory_path: str, number_of_overlaps: str = "1-12",
             model_input_size: int = 200, epochs: int = 10, batch_size: int = 1000,
             required_test_accuracy_pin: float = 0.88,
             dataset_files:int = 100, dataset_records_per_file: int = 15000) -> None:
    """Train the models with the data stored in the output path.

    Parameters
    ----------
    output_directory_path: str
        the directory the data folder is in that was created during the execution of create_dataset.py
    number_of_overlaps: str
        defining the range of overlaps as a string
    required_test_accuracy_pin: float
        is the accuracy the model of the pin has to achieve during training. If the accuracy is not achieved it tries again with a new model instance
    model_input_size: int
        the size of the input the model should be trained with
    dataset_files: int
        the amount of files that should be randomly sampled from the availabe data for training
    dataset_records_per_file: int
        amount of records from each file that is used for training

    Returns
    -------
    None

    """
    # Path configurations for data storage and model saving
    output_directory = output_directory_path
    data_directory = os.path.join(output_directory, "Data") 
    npy_data_directory = os.path.join(data_directory, "3_data_npy_train")
    npy_data_directory = os.path.join(npy_data_directory, number_of_overlaps)
    models_directory = os.path.join(data_directory, "models")
    pin_model_directory =os.path.join(models_directory,f"models_seq_{model_input_size}")

    # Ensure the model directory exists, create it if not
    os.makedirs(pin_model_directory, exist_ok=True)
    os.makedirs(models_directory, exist_ok=True)

    # Loading Filelist of Training-data
    filelist = os.listdir(npy_data_directory)
    filelist = [(x,y) for x in filelist if '_x_' in x for y in filelist if x.split('_')[0] == y.split('_')[0] and "_y_" in y]
    filelist.sort()

    # if possible use gpu instead of cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Taining is using GPU instead of CPU")

    pinwheel_sizes = [26]

    # Cumulative sizes used for iterating over segments
    cumulative_sizes = np.cumsum([0] + pinwheel_sizes)

    model_accuracies = []

    # Loop over each sequence length
    seq_len = model_input_size  # Define sequence length
    # Iterate over each wheel
    for wheel,(start_pin,end_pin) in enumerate(zip(cumulative_sizes, cumulative_sizes[1:])):

        # Iterate over each pin in the current wheel
        for pin in range(start_pin, end_pin):
            progress_data = []
            while True:
                model_pin = Model(input_length=seq_len, num_filters=100,
                                        depth=5, num_outputs=1)
                model_pin.to(device)

                dataset = LearnerDataset(model_input_size, npy_data_directory, pin,
                                        filelist, device, number_of_files=dataset_files,
                                        number_of_records_per_file=dataset_records_per_file)
                learner = Learner(model_pin, dataset)

                # model training
                learner.fit(batch_size, epochs, True, device=device)

                # model evaluation
                test_loss_pin, test_accuracy_pin = learner.evaluate()
                progress_data.append(test_accuracy_pin)

                print(f"Seq Length {seq_len}, Wheel {wheel}, Pin {pin}: Test Loss: {test_loss_pin}, Test Accuracy: {test_accuracy_pin}")

                # retraining for pins vs low accuracy
                if test_accuracy_pin > required_test_accuracy_pin:

                    # Record the model's accuracy
                    model_accuracies.append({
                        'Sequence Length': seq_len,
                        'Wheel Number': wheel,
                        'Pin Number': pin,
                        'Accuracy': test_accuracy_pin
                    })

                    # Save the final model
                    torch.save(model_pin, pin_model_directory + f'/best_model_wheel_{wheel}_pin_{pin}.pth')

                    # Clear the pytorch session and collect garbage to free memory
                    torch.cuda.empty_cache()
                    gc.collect()
                    break

    # Convert the list of accuracies to a DataFrame
    accuracy_df = pd.DataFrame(model_accuracies)
    # create new directory for the model_accuracies
    accuracies_directory = os.path.join(output_directory, "model_accuracies")
    os.mkdir(accuracies_directory)
    csv_file = os.path.join(accuracies_directory, "out.csv")
    # Save the DataFrame to an csv file
    accuracy_df.to_csv(csv_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_folder_path", type=str, help="path to the folder the data folder was created in during the create_dataset.py")
    args = parser.parse_args()
    training(args.output_folder_path, required_test_accuracy_pin=0.5)

