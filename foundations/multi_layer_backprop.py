import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        z1 = np.dot(W1,x) + b1
        a1 = np.maximum(z1, 0)
        z2 = np.dot(W2, a1) + b2
        output = {}
        output['loss'] = np.round(np.mean((z2-y_true)**2), 4)
        output['dW2'] = np.round(np.outer(2*(z2-y_true)/len(z2), a1),4)
        db2 = 2*(z2-y_true)/len(z2)
        output['db2'] = np.round(db2,4)
        db1 = np.dot(db2, W2)*(z1>0)
        output['db1'] = np.round(np.dot(db2, W2)*(z1>0), 4)
        output['dW1'] = np.round(np.outer(db1.T, x), 4)

        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        return output
