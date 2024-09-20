import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
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
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, batch_first= True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.embedding_layer = nn.Embedding(num_embeddings=27,
                                             embedding_dim=embedding_dim, padding_idx=26)
        self.linear_layer1 = nn.Linear(500, output_size)
        self.linear_layer_lugs = nn.Linear(500, 7)
        self.linear_layer2 = nn.Linear(embedding_dim, 1)
        self.linear_layer2_lugs = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.output_pins = nn.Linear(138, 131)
        self.output_lugs = nn.Linear(138, 7)
        self.pe = PositionalEncoding(d_model=embedding_dim, max_seq_length=500)

    def forward(self, x, inputsize):
        # Emmbedinglayer Batchsize x sequencelength -> batchsize
        padding = nn.ConstantPad1d((0, 500 - inputsize), 26)
        out = padding(x)
        out = self.embedding_layer(out)
        out = self.pe(out)
        out = self.encoder(out)
        out = torch.transpose(out, 1, 2)
        lugs = self.linear_layer_lugs(out)
        out = self.linear_layer1(out)
        out = torch.transpose(out, 1, 2)
        lugs = torch.transpose(lugs, 1, 2)
        out = self.linear_layer2(out)
        lugs = self.linear_layer2_lugs(lugs)

        #out = self.relu(out)
        lugs = self.relu(lugs)

        output = torch.cat((out,lugs),1)
        output = torch.transpose(output, 1, 2)
        out = self.output_pins(output)
        output = self.relu(output)
        lugs = self.output_lugs(output)
        out = torch.transpose(out, 1, 2)
        lugs = torch.transpose(lugs, 1, 2)

        out = self.sigmoid(out)
        lugs = self.relu(lugs)
        output = torch.cat((out,lugs),1)
        return output
