import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )
        self.fc1 = nn.Linear(in_features=120, out_features=50),  # Modify here to meet the requirement of feature extraction
        self.fc2 = nn.Linear(in_features=120, out_features=num_classes),

        ### Original LeNet5 backbone ###
        # self.fc_layers = nn.Sequential(
        #     nn.Linear(in_features=120, out_features=84),  Modify here to meet the requirement of feature extraction
        #     nn.Tanh(),
        #     nn.Linear(in_features=84, out_features=num_classes)
        # )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.Tanh()(x)
        x = self.fc2(x)
        return x

    def feature_extraction(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return nn.Tanh()(x)


