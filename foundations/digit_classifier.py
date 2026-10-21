import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        l1 = nn.Linear(784,512)
        l2 = nn.ReLU()
        l3 = nn.Dropout(0.2)
        l4 = nn.Linear(512, 10)
        self.layers = [l1,l2,l3,l4]

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        y = images
        for l in self.layers: 
            y = l(y)
        y = torch.round(torch.sigmoid(y), decimals=4)
        return y
