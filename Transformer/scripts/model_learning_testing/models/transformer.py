import torch
from torch import nn


class OutputNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(175, 256))
        self.fc_layer1 = nn.Linear(26, 2048)
        self.fc_layer2 = nn.Linear(2048, 26)
        self.relu = nn.ReLU()

    def forward(self, input):
        out = self.conv_layer(input.unsqueeze(1))
        out = self.relu(out.squeeze())
        out = self.fc_layer1(out)
        out = self.relu(out)
        out = self.fc_layer2(out)
        return out


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
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(175, 256))
        self.out_network = OutputNetwork()


    def forward(self, x):
        # Emmbedinglayer Batchsize x sequencelength -> batchsize 
        out = self.embedding_layer(x)
        #print(out.shape)
        #out = x.unsqueeze(2)
        out = self.encoder(out)
        #out = torch.transpose(out, 1, 2)
        #out = self.linear_layer1(out)
        #out = torch.transpose(out, 1, 2)
        #out = torch.mean(out, 2)
        out = self.out_network(out)
        #out = self.conv_layer(out.unsqueeze(1))
        #out = self.linear_layer1(out)
        #print(out.shape)
        #out = torch.transpose(out, 1, 2)
        out = self.sigmoid(out)
        return out
