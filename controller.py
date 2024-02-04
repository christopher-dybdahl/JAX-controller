import jax.numpy as jnp


class Controller:
    def __init__(self, controller_function, initial_parameters, learning_rate):
        self.controller_function = controller_function
        self.initial_parameters = initial_parameters
        self.parameters = initial_parameters
        self.learning_rate = learning_rate
        self.error_history = jnp.array([])

    def get_parameters(self):
        return self.parameters

    def get_error_history(self):
        return self.error_history

    def reset_parameters(self):
        self.parameters = self.initial_parameters

    def reset_error_history(self):
        self.error_history = jnp.array([])

    def update_params(self, gradients):
        # Gradient descent
        self.parameters -= jnp.multiply(self.learning_rate, gradients)

    def update(self, parameters, error):
        self.error_history = jnp.append(self.error_history, error)
        U = self.controller_function(self.error_history, parameters)
        return U
