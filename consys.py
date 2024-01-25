from jax import jit, value_and_grad

from helpers import noise


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
        self.error_history = []
        self.params_history = []
        self.mse_history = []
        self.controller.reset_parameters()
        self.reset_controller_plant()

    def reset_controller_plant(self):
        self.controller.reset_error_history()
        self.plant.reset_state()

    def simulate(self, epochs, timesteps):
        self.epochs = epochs
        self.timesteps = timesteps

        # Reset controller parameters
        self.controller.reset_parameters()

        # Initialize gradient function with jax
        gradient_function = jit(value_and_grad(self.loss_function, argnums=2))

        for epoch in range(self.epochs):
            # Reset other controller variables and plant state
            self.reset_controller_plant()

            # Generate noise
            D = noise()

            error = None
            # TODO: How to handle first error

            for timestep in range(self.timesteps):
                # Update controller and plant
                U = self.controller.update(error)
                Y = self.plant.update(U, D)
                error = self.T - Y

                # Record error in history
                self.error_history.append(error)

            # Declare som variables for readability
            model_function = self.controller.controller_function
            X = self.error_history
            w = self.controller.controller_params
            y_true = self.T

            # Compute MSE
            mse = self.loss_function(model_function, X, w, self.T)
            self.mse_history.append(mse)

            # Compute gradients
            avg_error, gradients = gradient_function(model_function, X, w, y_true)

            # Update controller parameters using gradient descent
            self.controller.update_params(gradients)
            self.params_history.append(self.controller.params)

        return U, Y, self.mse_history, self.params_history
