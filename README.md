# Transformer-Based Cryptoanalysis of the Hagelin M-209 Cipher Machine

## Overview
First part of this project is a conversion of the Github project [M209KnownPlaintextAttackML](https://github.com/CrypToolProject/M209KnownPlaintextAttackML) from tensorflow to pytorch.
Second Part is implementing a transformer-based model and test its performance for an KPA-Attack against the Hagelin M-209.

## How the Code is Organized
The m209 package contains the implementation of the hagelin m209 cipher.
In Dataset_creation are all files necessary to create the datasets for training and testing models for the given problem.
In the ConvoluttionalNN folder you can find the pytorch conversion with executable files for learning and testing the models. 
In the transformer folder are all files used to train and test a transformer-based model.
The pyproject.toml contains the settings for ruff

## External Encryption Tool
For encrypting data, we use an external M-209 implementation found in Brian Neal's M-209 GitHub Repository. This tool is utilized for accurately simulating the encryption process as performed by the actual Hagelin M-209 machine. The original repository is accessible [here](https://github.com/gremmie/m209). Note that for data organizational reasons, we slightly modified some of this code. Therefore, for correct performance, please use the version available in the folder "m209 Brian Neal" of our repository.

## External Dataset Creation Tool
To create a dataset for training and testing the model we use the tools provided in the Github project [M209KnownPlaintextAttackML](https://github.com/CrypToolProject/M209KnownPlaintextAttackML). We slightly modified the files, so that the data contains the 27 different lug_pairs used to encrypt the text.

## Getting Started

### Setup:
1. Install miniconda
2. Create conda environment with the environment file in root of repository:  
```bash
    conda env create -f environment.yml
```
3. Install the m209 package:
    - activate the created environment 
    - switch to the m209 Brian Neal directory
    - execute command: ```pip install .```

**Create dataset**: Run the script *create_data_set.py* in the Dataset_creation folder. You have to give a path to the folder you want your data to be generated in.

### ConvolutionalNN

**Learning a model** Run *train_model.py* in the *ConvolutionalNN* folder. This will train the model that is specified in the *model_learning_testing* package. It creates and saves the models in the data folder.

**Testing model** Run *test_model.py* in the *ConvolutionalNN* folder. This will test the models you generated in the previous step and then output some plots and csv files.


### Transformer

**Learning a model** Run *train_model.py* in the *Transformer* folder. It creates and saves the models in the data folder. You have to provide the path to the data folder and the name of the file(without *.py*) that contains a model named encoder. This model has to be in the folder *Transformer\scripts\model_learning_testing\models*. That way you can have multiple models in that folder and train and test them by providing the name of the file.

**Testing model** Run *test_model.py* in the *Transformer* folder. This will test the models you generated in the previous step and then output some plots and csv files, in the folder the path points to.
You can run *test_model_multiple_input_lengths.py* to test the model on different input lenghts. This does only work if the model is trained with a variable input length.