import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        h = x
        for i in range(len(weights)-1):
            w = weights[i]
            b = biases[i]
            h = np.dot(h, w) + b
            h = h*(h>0)
        return np.round(np.dot(h,weights[-1])+biases[-1],5)
