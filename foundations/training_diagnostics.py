import torch
import torch.nn as nn
from typing import List, Dict


class Solution:
    @torch.no_grad()
    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        res = []
        output = x
        for layer in model: 
            print(layer)
            layer_metrics = {}
            output = layer.forward(output)
            if type(layer) == nn.Linear: 
                layer_metrics['mean'] = round(torch.mean(output).item(), 4)
                layer_metrics['std'] = round(torch.std(output).item(), 4)
                layer_metrics['dead_fraction'] = round((output <= 0.0).all(dim=0).float().mean().item(), 4)
                res.append(layer_metrics)
        return res
            

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        criterion = nn.MSELoss()
        loss = criterion(model(x), y)
        loss.backward() 
        res = []
        for layer in model: 
            layer_metrics = {}
            if type(layer) == nn.Linear: 
                layer_metrics['mean'] = round(torch.mean(layer.weight.grad).item(), 4) 
                layer_metrics['std'] = round(torch.std(layer.weight.grad).item(), 4)
                layer_metrics['norm'] = round(torch.norm(layer.weight.grad).item(), 4)
                res.append(layer_metrics)
        return res


    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for act in activation_stats: 
            if act['dead_fraction'] > 0.5: 
                return 'dead_neurons'
        for grad in gradient_stats: 
            if grad['norm'] > 1000:
                return 'exploding_gradients'
        for grad in gradient_stats: 
            if grad['norm'] < 1e-5: 
                return 'vanishing_gradients'
        return 'healthy'

