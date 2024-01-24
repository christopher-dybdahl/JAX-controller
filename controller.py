class Controller:
    def __init__(self, controller_function, initial_parameters):
        self.controller_function = controller_function  # Either classic or neural net
        self.initial_parameters = initial_parameters
        self.parameters = initial_parameters
        self.error_history = []

    def reset(self):
        self.parameters = self.initial_parameters
        self.error_history = []

    def update(self, error):
        self.error_history.append(error)
        U = self.controller_function(self.error_history, self.parameters)
        return U

# TODO: improve parameters using JAX
