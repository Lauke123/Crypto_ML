import math

import numpy as np
import torch
from torch.nn import Module


class ModelTester:
    '''
    Testingclass for testing the accuracy of a traiend model.
    The class can test a binary classification model.
    '''
    def __init__(self, model_list:list[Module], device, inputsize):
        self.model_list = model_list
        self.device = device
        self.inputsize = inputsize

    def sample_data(self, X, y, sample_size:int):
        '''Sample a fix number of inputs random from a dataset'''
        total_samples = len(X)
        indices = np.random.choice(total_samples, sample_size, replace=False)
        return X[indices], y[indices]

    def normalize_data(self, inputs, lables):

        if self.inputsize > inputs.shape[1]:
            raise UserWarning("Length too short")

        inputs = inputs[:,:self.inputsize]

        inputs = np.subtract(inputs ,65)
        inputs = np.array(inputs, dtype='float32')
        inputs = np.divide(inputs , 25)

        lables = np.array(lables, dtype='float32')

        return inputs, lables

    def compute_predictions(self,x, batchsize:int):
        """Compute model predictions for a given dataset."""
        all_predictions = []
        num_batches = math.ceil(len(x) / batchsize)
        for model in self.model_list:
            model_predictions = []
            # To avoid running out of memory, predictions are made in batches
            for i in range(num_batches):
                start = i * batchsize
                end = min((i + 1) * batchsize,len(x))
                model_predictions_batch = model.forward(torch.tensor(x[start:end],
                                                                    device=self.device)
                                                                    .unsqueeze(1))
                model_predictions_batch = torch.round(model_predictions_batch)
                model_predictions_batch = model_predictions_batch.detach().cpu().numpy()
                model_predictions.extend(model_predictions_batch.flatten())
            all_predictions.append(model_predictions)
        return np.array(all_predictions).T  # Transpose so that each row represents a sample

    def count_correct_predictions(self, all_predictions, y, X):
        '''Compare the predictions with the labels and count the correct ones'''
        correct_counts = []
        for i in range(len(X)):
            correct_count = 0
            for j, model in enumerate(self.model_list):
                if all_predictions[i][j] == y[i][int(model.myname.split('_')[-1].split('.')[0])]:
                    correct_count += 1
            correct_counts.append(correct_count)
        return correct_counts
