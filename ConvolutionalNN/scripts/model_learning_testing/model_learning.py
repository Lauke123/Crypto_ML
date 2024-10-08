import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .dataloading import load_partial_data


class LearnerDataset(Dataset):
    """ 
    Dataset class to use the Dataloader in Learnerclass. The class imports
    data from a given path and splits it into test and train tensors.
    """

    def __init__(self,
                 inputsize:int,
                 data_path:str,
                 pin:int,
                 filelist,
                 device,
                 number_of_records_per_file: int = 15000,
                 number_of_files: int = 100):

        # load the data for training and evaluation of the model
        x, y = load_partial_data(count=number_of_files,records_per_file=number_of_records_per_file,
                                 filelist=filelist, path_data=data_path, inputsize=inputsize)
        targets = y[:, pin]
        # split the data into training and test data
        x_train, x_test, y_train, y_test = train_test_split(x, targets, test_size=0.2, random_state=17)
        # this split represents the validation_split parameter in the tensorflow fit function.
        x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)

        # test shape of training data, adding an extra dimension so the channel has a dimension in the tensor. 
        # The conv layer in the model expects a channel dimension with size = 1
        self.inputs_train = torch.tensor(x_train, device=device).unsqueeze(1)
        self.inputs_test = torch.tensor(x_test, device=device).unsqueeze(1)
        self.lable_train = torch.tensor(y_train, device=device)
        self.label_test = torch.tensor(y_test, device=device)
        self.len_train = self.inputs_train.shape[0]
        self.len_test = self.inputs_test.shape[0]

        # Bool to switch between using the training or testing dataset
        self.test_set = False

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



class Learner:
    ''' 
    Learningclass for fitting and evaluate models.
    The class takes a binary classification model and can fit it on a dataset
    '''
    def __init__(self, model:torch.nn.Module,
                 dataset:LearnerDataset,
                 learningrate: float = 0.001,
                 l2_kernel_regularizer: bool = True,
                 l2_kernel_reg_parameter: float = 0.0002):
        self.l2_kernel_regularizer = l2_kernel_regularizer
        self.l2_kernel_reg_parameter = l2_kernel_reg_parameter
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
                    prediction = self.model.forward(inputs.to(device))
                    prediction = torch.flatten(prediction)
                    loss = self.criterion(prediction, labels)
                    # used to calculate and add the l2 regularizer for all the conv layer weights, like in the original tensorflow implementation
                    if self.l2_kernel_regularizer:
                        conv_weight = [p for name, p in self.model.named_parameters() if "conv" in name and "weight" in name]
                        for weight in conv_weight:
                            loss += torch.norm(weight, p=2)**2 * self.l2_kernel_reg_parameter
                    # optimze the model with backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    def evaluate(self, batchsize=1000):
        correct_predictions = 0
        # Use the testdata of dataset
        self.dataset.set_to_testset()
        len_testset = self.dataset.__len__()
        with torch.no_grad():
            # creating set of data batches with dataloader
            dataloader = DataLoader(batch_size=batchsize, dataset=self.dataset, shuffle=False)
            for _, (inputs, labels) in enumerate(dataloader):
                # compute the predictions from the testset
                eval_pred = self.model.forward(inputs)
                eval_pred = torch.flatten(eval_pred)
                loss = self.criterion(eval_pred, labels)
                # used to calculate and add the l2 regularizer for all the conv layer weights, like in the original tensorflow implementation
                if self.l2_kernel_regularizer:
                    conv_weight = [p for name, p in self.model.named_parameters() if "conv" in name and "weight" in name]
                    for weight in conv_weight:
                        loss += torch.norm(weight, p=2)**2 * self.l2_kernel_reg_parameter
                eval_pred = torch.round(eval_pred)

                # count the number of correct predictions
                for i in range(len(labels)):
                    if labels[i] == eval_pred[i]:
                        correct_predictions +=1

        accuracy = correct_predictions / len_testset
        # Return back to default
        self.dataset.set_to_trainset()
        return loss, accuracy






