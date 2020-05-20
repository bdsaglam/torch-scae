import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationProbe(nn.Module):
    def __init__(self, dim_in, n_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_in, n_classes),
            nn.Softmax(-1),
        )
        self.n_classes = n_classes

    def forward(self, x, label):  # (B, dim_in), (B, )
        predicted_prob = self.network(x)
        xe_loss = F.cross_entropy(input=predicted_prob, target=label)
        accuracy = (torch.argmax(predicted_prob, 1) == label).float().mean()
        return xe_loss, accuracy
