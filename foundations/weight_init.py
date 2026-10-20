import torch
import torch.nn as nn
import math
from typing import List
import numpy as np


class Solution:
    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2 / (fan_in + fan_out))
        W = torch.randn(fan_out, fan_in) * std
        return [[round(v, 4) for v in row] for row in W.tolist()]

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2 / fan_in)
        W = torch.randn(fan_out, fan_in) * std
        return [[round(v, 4) for v in row] for row in W.tolist()]

    def check_activations(self, num_layers, input_dim, hidden_dim, init_type) -> List[float]:
        torch.manual_seed(0)
        res = []
        weights = []
        for i in range(num_layers):
            fan_in = input_dim if i == 0 else hidden_dim
            if init_type == 'xavier':
                std = math.sqrt(2.0 / (fan_in + hidden_dim))
            elif init_type == 'kaiming':
                std = math.sqrt(2.0 / fan_in)
            else:
                std = 1.0
            W = torch.randn(hidden_dim, fan_in) * std
            weights.append(W)
        x = torch.randn(1, input_dim)
        for W in weights:
            x = torch.relu(x @ W.T)
            res.append(round(x.std().item(), 2))
        return res