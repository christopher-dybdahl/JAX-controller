class Plant(object):
    def __init__(self, state_function):
        self.state_function = state_function  # Function either function for H or P_1
        self.state = None  # State either H or P_1

    def reset(self):
        self.state = None

    def update(self, U, D):
        self.state = self.state_function(U, D)
        return self.state

    # TODO: constrain q_1 and q_2 in Cournot
