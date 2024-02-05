import jax.numpy as jnp

from controller import Controller


class Classic_controller(Controller):
    def __init__(self, initial_parameters, input_processor, learning_rate):
        super().__init__(initial_parameters)
        self.input_processor = input_processor
        self.learning_rate = learning_rate

    def update_params(self, gradients):
        # Gradient descent
        self.parameters -= jnp.multiply(self.learning_rate, gradients)

    def update(self, parameters, error):
        # Record error
        self.error_history = jnp.append(self.error_history, error)

        # Process input and compute output
        U = jnp.dot(parameters, self.input_processor(self.error_history))
        return U
