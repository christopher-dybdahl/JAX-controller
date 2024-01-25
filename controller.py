import jax
import numpy as np


class Controller:
    def __init__(self, controller_function, initial_parameters, learning_rate):
        self.controller_function = controller_function  # Either classic or neural net
        self.initial_parameters = initial_parameters
        self.parameters = initial_parameters
        self.learning_rate = learning_rate
        self.error_history = []

    def reset_parameters(self):
        self.parameters = self.initial_parameters

    def reset_error_history(self):
        self.error_history = []

    def update_params(self, gradients):
        self.parameters -= self.learning_rate * gradients

    def update(self, error):
        self.error_history.append(error)
        U = self.controller_function(self.error_history, self.parameters)
        return U
