import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, (4, 9), 1)
        self.conv2 = nn.Conv1d(1, 1, 3, 1)
        self.fc = nn.Linear(47, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return nn.Sigmoid()(x)