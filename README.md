# Known Plaintext Attack against Hagelin M-209 Cipher Machine using Aritificial Neural Networks

## Overview
This project is a conversion of the original [project](https://github.com/CrypToolProject/M209KnownPlaintextAttackML) from tensorflow to pytorch.
This project is under development. Currently its getting tested if after the conversion it still achieves the same performance as the original implementation with tensorflow.
In the future we will add new models to try to achieve a better performance and then extend the code so it predicts the complete pin settings(+ lug settings) and not just the ones for the first wheel


## How the Code is Organized
The m209 package contains the implementation of the hagelin m209 cipher.
In the scripts folder are executable files for creating the dataset, learning and testing models and packages that are necessary for the scripts to run.
The pyproject.toml contains the settings for ruff

## External Encryption Tool

For encrypting data, we use an external M-209 implementation found in Brian Neal's M-209 GitHub Repository. This tool is utilized for accurately simulating the encryption process as performed by the actual Hagelin M-209 machine. The original repository is accessible [here](https://github.com/gremmie/m209). Note that for data organizational reasons, we slightly modified some of this code. Therefore, for correct performance, please use the version available in the folder "m209 Brian Neal" of our repository.

## Getting Started

**Setup**:
1. Install miniconda
2. Create conda environment with the environment file in root of repository:  
```bash
    conda env create -f environment.yml
```
3. Install the m209 package:
    - activate the created environment 
    - switch to the m209 Brian Neal directory
    - execute command: ```pip install .```

**Create dataset**: Run the script *create_data_set.py* in the scripts folder. You have to give a path to the folder you want your data to be generated in.

**Learning a model** Run *train_model.py* with the path to the folder the data was created in. This will train the model that is specified in the *model_learning_testing package*. It creates and saves the models in the data folder.
Currently if you would want to test a different neural network you have to replace the code in the Model class.

**Testing model** Run *test_model.py* with path to the folder the data was created in. This will test the models you generated in the previous step and then output some plots and csv files, in the folder the path points to.
