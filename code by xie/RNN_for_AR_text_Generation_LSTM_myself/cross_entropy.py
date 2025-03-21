import torch

class CustomCrossEntropyLoss:
    def forward(self, predictions, targets):
        batch_size = predictions.shape[0]
        log_likelihood = -torch.log(predictions[range(batch_size), targets])
        loss = torch.sum(log_likelihood) / batch_size
        self.predictions = predictions
        self.targets = targets
        return loss

    def backward(self):
        batch_size = self.predictions.shape[0]
        grad = self.predictions.clone()
        grad[range(batch_size), self.targets] -= 1
        grad /= batch_size
        return grad
