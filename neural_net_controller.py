import jax.numpy as jnp
import numpy as np

from controller import Controller


class Neural_net_controller(Controller):
    def __init__(self, input_processor, learning_rate, layers, activation_functions):
        initial_parameters = self.gen_jaxnet_params(layers)
        super().__init__(initial_parameters)
        self.input_processor = input_processor
        self.learning_rate = learning_rate
        self.layers = layers
        self.activation_functions = activation_functions

    def update_params(self, gradients):
        parameters = self.get_parameters()

        # Update parameters
        self.parameters = [(w - self.learning_rate * dw, b - self.learning_rate * db) for (w, b), (dw, db) in
                           zip(parameters, gradients)]

    def update(self, parameters, error):
        self.error_history = jnp.append(self.error_history, error)
        features = self.input_processor(self.error_history)

        # Neural network forwarding
        activations = features
        for (weights, biases), fun in zip(parameters, self.activation_functions):
            activations = fun(jnp.dot(activations, weights) + biases)
        return activations

    def gen_jaxnet_params(self, layers):
        # Initializing weights
        sender = layers[0]
        parameters = []
        for receiver in layers[1:]:
            weights = jnp.asarray(np.random.uniform(-.1, .1, (sender, receiver)))
            biases = jnp.asarray(np.random.uniform(-.1, .1, (1, receiver)))
            sender = receiver
            parameters.append([weights, biases])

        return parameters
