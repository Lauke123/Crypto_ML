import numpy as np
import torch
from torch.nn import Module


class ModelTester:
    def __init__(self, model_list:list[Module], device):
        self.model_list = model_list
        self.device = device

    def sample_data(self, X, y, sample_size:int):
        total_samples = len(X)
        indices = np.random.choice(total_samples, sample_size, replace=False)
        return X[indices], y[indices]

    def compute_predictions(self,x):
        """Compute model predictions for a given dataset."""
        all_predictions = []

        for model in self.model_list:
            model_predictions = []
            for i in range(100):
                model_predictions_batch = model.forward(torch.tensor(x[int(len(x)*float(i)/100):int(len(x)*float(i+1)/100)], device=self.device).unsqueeze(1))  
                model_predictions_batch = torch.round(model_predictions_batch)
                model_predictions_batch = model_predictions_batch.detach().cpu().numpy()
                model_predictions.extend(model_predictions_batch.flatten())
            all_predictions.append(model_predictions)
        return np.array(all_predictions).T  # Transpose so that each row represents a sample

    def count_correct_predictions(self, all_predictions, y, X):
        correct_counts = []
        for i in range(len(X)):
            correct_count = 0
            for j, model in enumerate(self.model_list):
                if all_predictions[i][j] == y[i][int(model.myname.split('_')[-1].split('.')[0])]:
                    correct_count += 1
            correct_counts.append(correct_count)
        return correct_counts
