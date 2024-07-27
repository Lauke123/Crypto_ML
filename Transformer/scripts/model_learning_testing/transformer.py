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

    def __init__(self, embedding_dim:int) -> None:
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first= True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.embedding_layer = nn.Embedding(num_embeddings=28,
                                             embedding_dim=256)
        self.linear_layer1 = nn.Linear(200, 26)
        self.linear_layer = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # Emmbedinglayer Batchsize x sequencelength -> batchsize 
        out = self.embedding_layer(x)
        print(out.shape)
        #out = x.unsqueeze(2)
        out = self.encoder(out)
        #out = torch.transpose(out, 1, 2)
        #out = self.linear_layer1(out)
        #out = torch.transpose(out, 1, 2)
        out = self.linear_layer(out)
        print(out.shape)
        #out = torch.transpose(out, 1, 2)
        out = self.sigmoid(out[:,:26,:])
        return out
