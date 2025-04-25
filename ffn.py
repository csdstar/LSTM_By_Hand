import torch.nn as nn


class FFN_in(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class FFN_out(nn.Module):
    def __init__(self, input_dim=512, output_dim=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
