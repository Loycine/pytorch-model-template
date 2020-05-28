import torch.nn as nn


class ToyNet(nn.Module):
    def __init__(self, output_dim: int):
        super(ToyNet, self).__init__()
        self.features = nn.Sequential(
            # define the extracting network here
            nn.Linear(2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            # define the classifier network here
            nn.Linear(output_dim, 2),
        )

    def forward(self, x):
        # define the forward function here
        x = self.features(x)
        x = self.classifier(x)
        return x
