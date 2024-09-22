import argparse
import gc
import os
import sys
import importlib
import pandas as pd
import torch
from model_learning_testing.model_learning import Learner, LearnerDataset


def training(output_directory_path: str, transformer_file_name:str, number_of_overlaps: str = "1-12",
             model_input_size: int = 200, wheelsize: int = 26, epochs: int = 10, batch_size: int = 100,
             required_test_accuracy_pin: float = 0.88,
             dataset_files:int = 100, dataset_records_per_file: int = 25000, lug_training:bool = False) -> None:
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
    transformer_file_name: str
        path to a file where a transformer class named Encoder is defined

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
    pin_model_directory =os.path.join(models_directory,f"transformer/{transformer_file_name}")

    # Ensure the model directory exists, create it if not
    os.makedirs(pin_model_directory, exist_ok=True)
    os.makedirs(models_directory, exist_ok=True)

    # initialize transformer model
    transformer = importlib.import_module("." + transformer_file_name, 'model_learning_testing.models')
    my_class = getattr(transformer, 'Encoder')
    model = my_class(model_input_size, output_size=wheelsize)

    # Loading Filelist of Training-data
    filelist = os.listdir(npy_data_directory)
    filelist = [(x,y,z) for x in filelist if '_x_' in x for y in filelist if x.split('_')[0] == y.split('_')[0] and "_y_ALL_" in y for z in filelist if y.split('_')[0] == z.split('_')[0] and "_y_lugs" in z]
    filelist.sort()

    # if possible use gpu instead of cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Taining is using GPU instead of CPU")

    # amount of pins on the first wheel of the hagelin m-209
    pinwheel_size = wheelsize

    model_accuracies = []



    while True:
        model.to(device)

        dataset = LearnerDataset(model_input_size, npy_data_directory, pinwheel_size,
                                filelist, device, number_of_files=dataset_files,
                                number_of_records_per_file=dataset_records_per_file, lug_training=lug_training)
        learner = Learner(model, dataset, learningrate=0.001)

        # model training
        learner.fit(batch_size, epochs, True, device=device)

        # model evaluation
        test_loss, test_accuracy = learner.evaluate(batchsize=batch_size)

        print(f"Seq Length {model_input_size}, Wheel 1,  Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        # retraining for pins vs low accuracy
        if test_accuracy > required_test_accuracy_pin:

            # Record the model's accuracy
            model_accuracies.append({
                'Sequence Length': model_input_size,
                'Wheel Number': 1,
                'Accuracy': test_accuracy
            })

            # Save the final model
            torch.save(model, pin_model_directory + f'/best_transformer.pth')

            # Clear the pytorch session and collect garbage to free memory
            torch.cuda.empty_cache()
            gc.collect()
            break

    # create new directory for the model_accuracies
    csv_file = os.path.join(pin_model_directory, "model_accuracies.csv")
    # Convert the list of accuracies to a DataFrame
    accuracy_df = pd.DataFrame(model_accuracies)
    # Save the DataFrame to an csv file
    accuracy_df.to_csv(csv_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_folder_path", type=str, help="path to the folder the data folder was created in during the create_dataset.py")
    parser.add_argument("transformer_file_name", type=str, help="name of file containing a transformer model, named 'Encoder'")
    parser.add_argument("-m", "--model_size", type=int, default=200, help="defines the size of the input layer of the model" )
    parser.add_argument("-w", "--wheel_size", type=int, default=26, help="defines how many pins should be predicted" )
    parser.add_argument("-l", "--lug_training", type=bool, default=False, help="enables learning the lugs settings aswell, this increases the target data size by 7" )
    args = parser.parse_args()
    # adjust the parameters for training if you want to apply some form of control to the training process
    training(args.output_folder_path, required_test_accuracy_pin=0.5,
             model_input_size=args.model_size, transformer_file_name=args.transformer_file_name, wheelsize= args.wheel_size, lug_training=args.lug_training)

