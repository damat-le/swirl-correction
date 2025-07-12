import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomModel(nn.Module):
    
    def __init__(self):
        super(RandomModel, self).__init__()

        self.linear = nn.Linear(1,1) # just to avoid empty model error

    def forward(self, x):
        """
        A dummy forward pass that returns a random tensor with the same shape as the input.
        """
        return torch.rand_like(x), None