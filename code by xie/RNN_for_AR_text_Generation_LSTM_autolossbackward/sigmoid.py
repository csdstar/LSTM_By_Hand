import torch
import torch.nn as nn

class CustomSigmoid(nn.Module):
    def forward(self, x):
        return 1 / (1 + torch.exp(-x))
