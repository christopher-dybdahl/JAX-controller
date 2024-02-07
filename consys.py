import matplotlib.pyplot as plt
from jax import jit, value_and_grad

from helpers import noise


def standard_eval(state):
    return state


class Consys:
    def __init__(self, plant, controller, T, loss_function, state_eval=standard_eval):
        self.plant = plant
        self.controller = controller
        self.T = T
        self.loss_function = loss_function
        self.state_eval = state_eval

        self.epochs = None
        self.timesteps = None

        self.params_history = None
        self.loss_history = None
        self.state_history = []

    def reset_controller_plant(self):
        self.controller.reset_error_history()
        self.plant.reset_state()

    def simulate(self, epochs, timesteps, noise_range, U_init):
        # Record epochs and timesteps
        self.epochs = epochs
        self.timesteps = timesteps

        # Reset controller parameters
        self.params_history = []
        self.loss_history = []
        self.controller.reset_parameters()
        self.reset_controller_plant()

        # Record initial parameters
        self.params_history.append(self.controller.get_parameters())

        # Initialize gradient function with jax
        gradient_function = jit(value_and_grad(self._run_one_epoch, argnums=0))

        for epoch in range(self.epochs):
            # Reset other controller variables and plant state
            self.reset_controller_plant()

            # Generate noise
            D = noise(noise_range, self.timesteps)

            # Compute value and gradients
            parameters = self.controller.get_parameters()
            loss, gradients = gradient_function(parameters, D, U_init)

            # Record error
            self.loss_history.append(loss)

            # Update controller parameters
            self.controller.update_params(gradients)

            # Record parameters
            self.params_history.append(self.controller.get_parameters())
            self.state_history.append(self.plant.get_state())

    def _run_one_epoch(self, parameters, D, U_init):

        U = U_init
        for _, D_t in zip(range(self.timesteps), D):
            # Update controller and plant, and evaluate error
            Y = self.plant.update(U, D_t)
            error_t = self.T - self.state_eval(Y)
            U = self.controller.update(parameters, error_t)

        loss = self.loss_function(self.controller.get_error_history())

        return loss

    def print_mse_history(self, title):
        plt.yscale("log")
        plt.plot(self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title(f'Learning Progression - {title}')
        plt.show()

    def print_parameter_history(self, title, legends=None):
        plt.plot(self.params_history)
        plt.xlabel('Epoch')
        plt.ylabel('Y')
        if legends is not None:
            plt.legend(legends)
        plt.title(f'Control Parameters - {title}')
        plt.show()

    def print_state_history(self):
        plt.plot(self.state_history)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title(f'Learning Progression - Plant State')
        plt.show()