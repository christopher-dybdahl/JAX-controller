import numpy as np


class Controller:
    def __init__(self, parameters):
        self.parameters = parameters
        self.gradient_function = None
        self.loss_function = None
        self.timesteps = None
        self.epochs = None
        self.error_history = []

    def initialize(self):
        self.error_history = []
        # TODO: Reset plant

    def simulate(self, epochs, timesteps, loss_function, gradient_function):
        self.epochs = epochs
        self.timesteps = timesteps
        self.loss_function = loss_function
        self.gradient_function = gradient_function

        for epoch in range(self.epochs):
            self.initialize()
            for timestep in range(self.timesteps):
                # TODO: find error
                error = None
                self._update(error)

        return None

    def _update(self, error):
        self.error_history.append(error)
        self.parameters = self.parameters - self.gradient_function(error)
        return None
