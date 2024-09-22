import torch
from torch import nn
import math
    
class Encoder(nn.Module):
    ''' 
    Transformer architecture using only the encoder part.
    Inputs are displacement values with variable sequence lengths.
    Outputs are the sequence of pins for one wheel
    It transforms the displacement values into embeddings and 
    uses the attention model to find correlations between each input.
    '''
    def __init__(self, input_size: int, embedding_dim:int = 512, output_size:int = 26) -> None:
        super().__init__()
        self.linear_layer1 = nn.Linear(500, output_size)
        self.linear_layer_lugs = nn.Linear(500, 22)
        self.linear_layer2 = nn.Linear(embedding_dim, 1)
        self.linear_layer2_lugs = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.output_pins = nn.Linear(153, 131)
        self.output_lugs = nn.Linear(153, 22)

    def forward(self, x, inputsize):
        # Emmbedinglayer Batchsize x sequencelength -> batchsize
        padding = nn.ConstantPad1d((0, 500 - inputsize), 26)
        out = padding(x).float()
        pins = self.linear_layer1(out)
        lugs = self.linear_layer_lugs(out)

        output = torch.cat((pins,lugs),1)
        pins = self.output_pins(output)
        lugs = self.output_lugs(output)

        pins = self.sigmoid(pins)
        lugs = self.relu(lugs)
        
        output = torch.cat((pins,lugs),1)
        output = output.unsqueeze(2)

        return output
