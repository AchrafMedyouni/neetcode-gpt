class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        old_value = init
        for i in range(iterations):
            new_value = old_value - learning_rate*2*old_value
            old_value = new_value
        return round(old_value, 5)
