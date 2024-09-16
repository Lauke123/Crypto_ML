import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random

from .dataloading import load_partial_data


class LearnerDataset(Dataset):
    ''' 
    Dataset class to use the Dataloader in Learnerclass. The class imports
    data from a given path and splits it into test and train tensors.
    '''
    def __init__(self,
                 inputsize:int,
                 data_path:str,
                 wheelsize:int,
                 filelist,
                 device,
                 number_of_records_per_file: int = 15000,
                 number_of_files: int = 100):

        x, y = load_partial_data(count=number_of_files,records_per_file=number_of_records_per_file,
                                 filelist=filelist, path_data=data_path, inputsize=inputsize)
        # Reshape data lables to one wheel
        targets = y[:, :wheelsize]

        X_train, X_test, y_train, y_test = train_test_split(x, targets, test_size=0.2, random_state=17)


        self.inputs_train = torch.tensor(X_train, device=device)
        self.inputs_test = torch.tensor(X_test, device=device)
        self.lable_train = torch.tensor(y_train, device=device)
        self.label_test = torch.tensor(y_test, device=device)
        self.len_train = self.inputs_train.shape[0]
        self.len_test = self.inputs_test.shape[0]
        self.inputsize = inputsize

        # Bool for using the dataset as testset or trainset
        # So the dataset could be uesed for traing and testing
        self.test_set = False

        self.wheelsize = wheelsize

    def __getitem__(self, index):
        if self.test_set:
            return self.inputs_test[index], self.label_test[index]
        return self.inputs_train[index], self.lable_train[index]

    def __len__(self):
        if self.test_set:
            return self.len_test
        return self.len_train

    def set_to_testset(self):
        self.test_set = True

    def set_to_trainset(self):
        self.test_set = False
    
    def get_wheelsize(self):
        return self.wheelsize
    
    def get_inputsize(self):
        return self.inputsize



class Learner:
    ''' 
    Learningclass for fitting and evaluate models.
    The class takes a binary classification model and can fit it on a dataset
    '''
    def __init__(self, model:torch.nn.Module,
                 dataset:LearnerDataset,
                 learningrate: float = 0.001):
        self.model = model
        self.criterion = nn.BCELoss()
        self.learningrate = learningrate
        self.dataset = dataset

    def fit(self, batchsize: int,
            epochs:int,
            shuffle:bool,
            device:torch.device):
        # creating set of data batches with dataloader
        dataloader = DataLoader(batch_size=batchsize, dataset=self.dataset, shuffle=shuffle)
        for epoch in tqdm(range(epochs), unit="epoch", desc="Progress of training"):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=(self.learningrate * (0.95 ** float(epoch))))
            for _, (inputs, labels) in enumerate(dataloader):
                    # compute predictions and loss from the trainset
                    input_length = self.dataset.get_inputsize()
                    if input_length == 500:
                        input_length = 10 * random.randint(3, self.dataset.get_inputsize()//10)
                        inputs = inputs[:,:input_length]
                    prediction = self.model.forward(inputs, input_length)
                    # Reshape labels to embedingsize
                    #For example: label = 1 -> label = 1,1,1...1,1 (dimensionsize:512)

                    prediction = torch.squeeze(prediction)

                    loss = self.criterion(prediction, labels)
                    
                    # optimze the model with backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    def evaluate(self, batchsize=1000):
        correct_predictions = 0
        # Use the testdata of dataset
        self.dataset.set_to_testset()
        len_testset = self.dataset.__len__() * self.dataset.get_wheelsize()
        with torch.no_grad():
            # creating set of data batches with dataloader
            dataloader = DataLoader(batch_size=batchsize, dataset=self.dataset, shuffle=False)
            for _, (inputs, labels) in enumerate(dataloader):
                # compute the predictions from the testset
                eval_pred = self.model.forward(inputs, self.dataset.get_inputsize())
                # the mean of each prediction embedding (is between 0 and 1) 
                # describes the prediction for one pin
                eval_pred = torch.flatten(eval_pred)
                labels = torch.flatten(labels)
                loss = self.criterion(eval_pred, labels)

                eval_pred = torch.round(eval_pred)

                # count the number of correct predictions
                for i in range(len(labels)):
                    if labels[i] == eval_pred[i]:
                        correct_predictions +=1

        accuracy = correct_predictions / len_testset
        # Return back to default
        self.dataset.set_to_trainset()
        return loss, accuracy






