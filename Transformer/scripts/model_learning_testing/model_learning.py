import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random

from .dataloading import load_partial_data


# custom loss for lugs predictions
class CustomLoss(nn.Module):
    def __init__(self, threshold=27, penalty_factor=2.0):
        super(CustomLoss, self).__init__()
        self.threshold = threshold
        self.penalty_factor = penalty_factor
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        # Base loss (MSE)
        base_loss = self.mse(predictions, targets)
        
        # Create a mask for values where predictions are greater than the threshold
        over_threshold = predictions > self.threshold
        
        # Calculate the additional penalty for over-prediction
        penalty = torch.abs(predictions - targets) * self.penalty_factor
        
        # Apply penalty only to predictions over the threshold
        penalty_loss = penalty * over_threshold.float()
        
        # Final loss is the base loss + the penalized loss
        total_loss = base_loss + penalty_loss.mean()
        
        return total_loss

# normalize the predicted amount of lugs on each bar so the the total sum is 54
def normalize_and_round(tensor, target_sum=54):
    # Get the current sum of each row
    current_sum = tensor.sum(dim=1, keepdim=True)
    
    # Calculate the scaling factor for each row
    scaling_factor = target_sum / current_sum
    
    # Normalize each row by scaling
    normalized_tensor = tensor * scaling_factor
    
    # Step 1: Floor the values to get integer parts
    int_tensor = torch.floor(normalized_tensor)
    
    # Step 2: Calculate the difference for each row (how much is needed to reach 54)
    diff = (target_sum - int_tensor.sum(dim=1)).int()
    
    # Step 3: Get the fractional parts of the normalized tensor
    fractional_part = normalized_tensor - int_tensor
    
    # Step 4: Sort indices by fractional parts in descending order for each row
    _, indices = torch.sort(fractional_part, descending=True, dim=1)
    
    # Step 5: Add 1 to the largest fractional values for each row until the sum equals 54
    for i in range(tensor.size(0)):  # Loop over each row
        for j in range(diff[i].item()):  # Add 1 to the top `diff[i]` indices in each row
            int_tensor[i, indices[i, j]] += 1

    return int_tensor


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
                 number_of_files: int = 100,
                 lug_training: bool = False):

        x, y = load_partial_data(count=number_of_files,records_per_file=number_of_records_per_file,
                                 filelist=filelist, path_data=data_path, inputsize=inputsize, lugs=lug_training)
        # Reshape data lables to one wheel
        target_length = wheelsize
        if lug_training:
            target_length += 7
        targets = y[:, :target_length]

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
        self.lug_training = lug_training

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
        self.criterion2 = CustomLoss()
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
                    prediction_pins = prediction[:, 0:self.dataset.wheelsize]
                    labels_pins = labels[:, 0:self.dataset.wheelsize]
                    loss = self.criterion(prediction_pins, labels_pins)
                    
                    if self.dataset.lug_training:
                        prediction_lugs = prediction[:, self.dataset.wheelsize:]
                        labels_lugs = labels[:, self.dataset.wheelsize:]
                        loss += self.criterion2(prediction_lugs, labels_lugs)
                        
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
                eval_pred = torch.squeeze(eval_pred)
                # split prediction and labels to pins, lugs
                prediction_pins = eval_pred[:, 0:self.dataset.wheelsize]
                labels_pins = labels[:, 0:self.dataset.wheelsize]
                loss = self.criterion(prediction_pins, labels_pins)
                
                if self.dataset.lug_training:
                    prediction_lugs = eval_pred[:, self.dataset.wheelsize:]
                    labels_lugs = labels[:, self.dataset.wheelsize:]
                    loss += self.criterion2(prediction_lugs, labels_lugs)
                    prediction_lugs = normalize_and_round(prediction_lugs)
                    prediction_lugs = torch.flatten(prediction_lugs)
                    labels_lugs = torch.flatten(labels_lugs)
                
                prediction_pins = torch.flatten(prediction_pins)
                labels_pins = torch.flatten(labels_pins)


                prediction_pins = torch.round(prediction_pins)

                # count the number of correct predictions of the pins
                for i in range(len(labels_pins)):
                    if labels_pins[i] == prediction_pins[i]:
                        correct_predictions +=1
                # count correct predictions of the lugs
                if self.dataset.lug_training:
                    abs_difference = torch.abs(prediction_lugs - labels_lugs)
                    #TODO write better evaluation of accuracy of lugs
                    average_difference = torch.mean(abs_difference).item()
                    print(average_difference)
        accuracy = correct_predictions / len_testset
        # Return back to default
        self.dataset.set_to_trainset()
        return loss, accuracy






