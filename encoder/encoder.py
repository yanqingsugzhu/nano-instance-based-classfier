import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, len_of_fv, len_of_DNA):
        super(Encoder, self).__init__()
        self.len_of_fv = len_of_fv
        self.len_of_DNA = len_of_DNA
        self.fc1 = nn.Linear(in_features=len_of_fv, out_features=25),
        self.fc2 = nn.Linear(in_features=25, out_features=4*len_of_DNA),

    def forward(self, x):
        x = fc1(x)
        x = fc2(x)
        x = nn.Softmax()(x)
        return x.reshape(-1, 4, self.len_of_DNA)
