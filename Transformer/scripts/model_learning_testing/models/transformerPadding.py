import torch
from torch import nn



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
                                             embedding_dim=embedding_dim, padding_idx=26)
        self.padding = nn.ConstantPad1d((0, 500 - input_size), 26)
        self.linear_layer1 = nn.Linear(500, 26)
        self.linear_layer2 = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # Emmbedinglayer Batchsize x sequencelength -> batchsize
        out = self.padding(x)
        print(out)
        out = self.embedding_layer(out)
        print(out)
        out = self.encoder(out)
        out = torch.transpose(out, 1, 2)
        out = self.linear_layer1(out)
        out = torch.transpose(out, 1, 2)
        out = self.linear_layer2(out)
        out = self.sigmoid(out)
        return out