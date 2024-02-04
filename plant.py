class Plant(object):
    def __init__(self, state_function, state_init):
        self.state_function = state_function
        self.state_init = state_init
        self.state = state_init

    def reset_state(self):
        self.state = self.state_init

    def update(self, U, D):
        Y = self.state_function(U, D, self.state)
        self.state = Y
        return Y
