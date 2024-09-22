import math

import numpy as np
import torch
from torch.nn import Module
from .model_learning import normalize_and_round


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
        lug_predictions = []
        num_batches = math.ceil(len(x) / batchsize)
        # To avoid running out of memory, predictions are made in batches
        for i in range(num_batches):
            start = i * batchsize
            end = min((i + 1) * batchsize,len(x))
            model_predictions_batch = self.model.forward(torch.tensor(x[start:end],
                                                                device=self.device),
                                                                inputsize)
            #model_predictions_batch = model_predictions_batch.squeeze()
            pin_predictions_batch = model_predictions_batch[:,:self.wheelsize]

            if self.lug_training:
                lug_prediction_batch = model_predictions_batch[:, self.wheelsize:]
                lug_prediction_batch = lug_prediction_batch.squeeze()
                lug_prediction_batch = normalize_and_round(lug_prediction_batch)
                lug_prediction_batch= lug_prediction_batch.detach().cpu().numpy()
                lug_predictions.extend(lug_prediction_batch)

            pin_predictions_batch = torch.round(pin_predictions_batch)
            pin_predictions_batch = pin_predictions_batch.detach().cpu().numpy()
            all_predictions.extend(pin_predictions_batch)
        return np.array(all_predictions), np.array(lug_predictions)  # Transpose so that each row represents a sample

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


    def test_avg_difference_lugs(self, lug_predictions, y)->list[list]:
        lug_targets = y[:,self.wheelsize:]
        print(lug_predictions.shape)
        print(lug_targets.shape)
        # calculate the mean of the 7 columns of tensor
        # Step 1: Compute the absolute difference element-wise
        abs_diff = np.abs(lug_predictions - lug_targets)

        split_columns = np.hsplit(abs_diff, 22)

        lug_position_values = [column.squeeze().tolist() for column in split_columns]
        
        return lug_position_values
