import math

import torch
from torch import nn


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
    def __init__(self, input_size: int, embedding_dim:int = 512) -> None:
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, batch_first= True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.embedding_layer = nn.Embedding(num_embeddings=28,
                                             embedding_dim=embedding_dim)
        self.positional_enc = PositionalEncoding(d_model=embedding_dim, max_seq_length=input_size)
        self.linear_layer1 = nn.Linear(200, 26)
        self.linear_layer2 = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # Emmbedinglayer Batchsize x sequencelength -> batchsize

        out = self.embedding_layer(x)
        out = self.positional_enc(out)
        out = self.encoder(out)
        out = torch.transpose(out, 1, 2)
        out = self.linear_layer1(out)
        out = torch.transpose(out, 1, 2)
        out = self.linear_layer2(out)
        out = self.sigmoid(out)
        return out



