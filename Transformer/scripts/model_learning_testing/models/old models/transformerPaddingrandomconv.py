import torch
from torch import nn

class OutputNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.pad = nn.ConstantPad2d((0, 0, 0, 20), 0)
        self.conv_layer = nn.Conv2d(1, 1, (20,1), dilation=(26,1))
        self.fc_layer1 = nn.Linear(512, 1)
        self.fc_layer2 = nn.Linear(2048, 26)
        self.relu = nn.ReLU()

    def forward(self, input):
        out = self.pad(input)
        out = self.conv_layer(out.unsqueeze(1))
        out = self.fc_layer1(out.squeeze())
        return out

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
        self.sigmoid = nn.Sigmoid()
        self.ouputnetwork = OutputNetwork()


    def forward(self, x, inputsize):
        # Emmbedinglayer Batchsize x sequencelength -> batchsize
        padding = nn.ConstantPad1d((0, 500 - inputsize), 26)
        out = padding(x)
        out = self.embedding_layer(out)
        out = self.encoder(out)
        out = self.ouputnetwork(out)
        out = self.sigmoid(out)
        return out
