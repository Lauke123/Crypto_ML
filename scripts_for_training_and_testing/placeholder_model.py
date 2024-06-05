import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple torch model to test the torch implementation
class Model(nn.Module):

    
    def __init__(self, input_length=100, num_filters=32, num_outputs=1, d1=512, d2=512, ks=5, depth=5, reg_param=0.0002, final_activation='sigmoid'):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_length,32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64,num_outputs)
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

if __name__ == "__main__":
    model = Model(input_length=200)
