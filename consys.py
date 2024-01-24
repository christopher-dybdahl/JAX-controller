from helpers import noise


class Consys:
    def __init__(self, plant, controller, T):
        self.plant = plant
        self.controller = controller
        self.T = T
        self.epochs = None
        self.timesteps = None

    def reset(self):
        self.plant.reset()
        self.controller.reset()

    def simulate(self, epochs, timesteps):
        self.epochs = epochs
        self.timesteps = timesteps

        for epoch in range(self.epochs):
            self.reset()
            D = noise()
            error = None  # TODO: How to handle first error
            # TODO: Implement visuals on parameters per epoch

            for timestep in range(self.timesteps):
                U = self.controller.update(error)
                Y = self.plant.update(U, D)
                error = self.T - Y
        return U, Y, error

# TODO: Implement visuals on MSE
