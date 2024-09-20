import math

import numpy as np
import torch
from torch.nn import Module


class ModelTester:
    '''
    Testingclass for testing the accuracy of a traiend model.
    The class can test a binary classification model.
    '''
    def __init__(self, model:Module, device, inputsize, wheelsize, lug_training:bool = False):
        self.model = model
        self.device = device
        self.inputsize = inputsize
        self.wheelsize = wheelsize
        self.lug_training = lug_training

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
        inputs = np.array(inputs, dtype='int32')

        lables = np.array(lables, dtype='float32')

        return inputs, lables

    def compute_predictions(self,x, batchsize:int, inputsize:int):
        """Compute model predictions for a given dataset."""
        all_predictions = []
        num_batches = math.ceil(len(x) / batchsize)
        # To avoid running out of memory, predictions are made in batches
        for i in range(num_batches):
            start = i * batchsize
            end = min((i + 1) * batchsize,len(x))
            model_predictions_batch = self.model.forward(torch.tensor(x[start:end],
                                                                device=self.device),
                                                                inputsize)
            model_predictions_batch = model_predictions_batch.squeeze()
            if self.lug_training:
                model_predictions_batch = model_predictions_batch[:,:self.wheelsize]
            model_predictions_batch = torch.round(model_predictions_batch)
            model_predictions_batch = model_predictions_batch.detach().cpu().numpy()
            all_predictions.extend(model_predictions_batch)
        return np.array(all_predictions)  # Transpose so that each row represents a sample

    def count_correct_predictions(self, all_predictions, y, X):
        '''Compare the predictions with the labels and count the correct ones'''
        correct_counts = []
        targets = y[:,:self.wheelsize]
        correct_count = 0
        for i in range(len(X)):
            correct_count = 0
            for j in range(targets.shape[1]):
                if all_predictions[i][j] == targets[i][j]:
                    correct_count += 1
            correct_counts.append(correct_count)
        return correct_counts

