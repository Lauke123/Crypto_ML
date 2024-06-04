import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import placeholder_model
from tqdm import tqdm


class LearnerDataset(Dataset):
    ''' 
    Dataset class to use the Dataloader in Learnerclass
    '''
    def __init__(self, inputs:torch.Tensor, lables:torch.Tensor):
        self.x = inputs
        self.y = lables
        self.len = inputs.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len



class Learner:
    ''' 
    Learningclass for fitting and evaluate models.
    The class takes a binary classification model and can fit it on a dataset
    '''
    def __init__(self, model: torch.nn.Module, learningrate=0.001):
        self.model = model
        self.criterion = nn.BCELoss()
        self.learningrate = learningrate

    

    def fit(self, batchsize: int, labels: torch.Tensor,  
            inputs:torch.Tensor, epochs:int, shuffle:bool):
        # creating set of data batches with dataloader
        dataset = LearnerDataset(inputs=inputs, lables=labels)
        dataloader = DataLoader(batch_size=batchsize, dataset=dataset, shuffle=shuffle)
        for epoch in tqdm(range(epochs), unit="epoch", desc="Progress of training"):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=(self.learningrate * (0.95 ** float(epoch))))
            for _, (inputs, labels) in enumerate(dataloader):
                    # compute predictions and loss from the trainset
                    prediction = self.model.forward(inputs)
                    prediction = torch.flatten(prediction)
                    loss = self.criterion(prediction, labels)
                    # optimze the model with backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    def evaluate(self, inputs:torch.Tensor, lables:torch.Tensor):
        correct_predictions = 0
        with torch.no_grad():
            # compute the predictions from the testset
            eval_pred = self.model.forward(inputs)
            eval_pred = torch.flatten(eval_pred)
            loss = self.criterion(eval_pred, lables)
            eval_pred = torch.round(eval_pred)

            # count the number of correct predictions
            for i in range(len(lables)):
                if lables[i] == eval_pred[i]:
                    correct_predictions +=1

        accuracy = correct_predictions / len(lables)

        return loss, accuracy






