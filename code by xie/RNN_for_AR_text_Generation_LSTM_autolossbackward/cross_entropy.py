import torch
import torch.nn as nn

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        log_probs = torch.log_softmax(y_pred, dim=1)
        loss = -log_probs[torch.arange(y_true.size(0)), y_true]
        return torch.mean(loss)
