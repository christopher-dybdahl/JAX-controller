from jax import jit, value_and_grad

from helpers import noise
import matplotlib.pyplot as plt
import jax.numpy as jnp


class Consys:
    def __init__(self, plant, controller, T, loss_function):
        self.plant = plant
        self.controller = controller
        self.T = T
        self.loss_function = loss_function
        self.error_history = None
        self.params_history = None
        self.mse_history = None
        self.epochs = None
        self.timesteps = None

    def reset(self):
        self.error_history = jnp.array([])
        self.params_history = []
        self.mse_history = []
        self.controller.reset_parameters()
        self.reset_controller_plant()

    def reset_controller_plant(self):
        self.controller.reset_error_history()
        self.plant.reset_state()

    def simulate(self, epochs, timesteps, noise_range, error_init):
        self.epochs = epochs
        self.timesteps = timesteps

        # Reset controller parameters
        self.reset()

        # Initialize gradient function with jax
        gradient_function = value_and_grad(self.loss_function, argnums=2)

        for epoch in range(self.epochs):
            # Reset other controller variables and plant state
            self.reset_controller_plant()

            # Generate noise
            D = noise(noise_range, self.timesteps)

            error = 0
            # TODO: How to handle first error

            for timestep, D_t in zip(range(self.timesteps), D):
                # Update controller and plant
                U = self.controller.update(error)
                Y = self.plant.update(U, D_t)
                error = self.T - Y

                # Record error in history
                self.error_history = jnp.append(self.error_history, error)

            # Declare some variables for readability
            model_function = self.controller.controller_function
            X = self.error_history
            w = self.controller.parameters
            y_true = self.T

            # Compute MSE
            mse = self.loss_function(model_function, X, w, self.T)
            self.mse_history.append(mse)

            # Compute gradients
            avg_error, gradients = gradient_function(model_function, X, w, y_true)

            # Update controller parameters using gradient descent
            self.controller.update_params(gradients)
            self.params_history.append(self.controller.parameters)
            print(self.params_history)
            # TODO: Check if correct MSE computing and gradient descent

        return U, Y

    def print_mse_history(self):
        plt.plot(self.mse_history)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Learning Progression')
        plt.legend()
        plt.show()

    def print_parameter_history(self):
        plt.plot(self.params_history)
        plt.xlabel('Epoch')
        plt.ylabel('Y')
        plt.title('Control Parameters')
        plt.show()
