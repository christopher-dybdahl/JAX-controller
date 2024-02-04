from abc import abstractmethod

import jax.numpy as jnp


class Controller:
    def __init__(self, initial_parameters):
        self.initial_parameters = initial_parameters
        self.parameters = initial_parameters
        self.error_history = None

    def get_parameters(self):
        return self.parameters

    def get_error_history(self):
        return self.error_history

    def reset_parameters(self):
        self.parameters = self.initial_parameters

    def reset_error_history(self):
        self.error_history = jnp.array([])

    @abstractmethod
    def update_params(self, gradients):
        pass

    @abstractmethod
    def update(self, parameters, error):
        pass
