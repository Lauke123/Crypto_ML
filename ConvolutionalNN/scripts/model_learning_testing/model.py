from torch import nn
from torchview import draw_graph


# a basic building block which can be included in other neural networks. It has two consecutive Conv layers and a skip connection from input to output
class ResidualBlock(nn.Module):
    def __init__(self, channels, stride=1, kernel_size=1, padding="same"):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, 
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, 
                               stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = out + x
        return out

# This represents the model of the original implementation in tensorflow
# the model does not include kernel regularisation like the model in tensorflow, it is manually calculated and added to the loss during the learning process
class Model(nn.Module):

    def __init__(self, input_length=100, num_filters=32, num_outputs=1, d1=512, d2=512, ks=5, depth=5, final_activation='sigmoid'):
        super(Model, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=1, padding="same", stride=1)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.relu_1 = nn.ReLU()
        self.res_net = nn.Sequential(*[ResidualBlock(channels=num_filters, kernel_size=ks, stride=1, padding="same") for i in range(depth)])
        self.flat1 = nn.Flatten()
        self.linear1 = nn.Linear(in_features=input_length * num_filters, out_features=d1)
        self.bn2 = nn.BatchNorm1d(d1)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=d1, out_features=d2)
        self.bn3 = nn.BatchNorm1d(d2)
        self.relu3 = nn.ReLU()
        self.linear3 = nn.Linear(d2, num_outputs)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn1(out)
        out = self.relu_1(out)
        out = self.res_net(out)
        out = self.flat1(out)
        out = self.linear1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.linear2(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.linear3(out)
        out = self.sigmoid(out)
        return out

if __name__ == "__main__":
    # get a visualisation of the model for better understanding and comparison
    model = Model(num_filters=100, input_length=200)
    batch_size = 10
    model_graph = draw_graph(model, input_size=(batch_size, 1, 200))
    graph = model_graph.visual_graph
    graph.render(filename="pytorch_graph")

