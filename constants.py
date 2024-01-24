import numpy as np

# Epoch and timesteps
epochs = 4
timesteps = 10

# Controller functions and parameters
# Generic Model of a standard 3-parameter PID Controller
# TODO: implement learning rate
classic_init_param = 0.1, 0.01, 0.001


def classic_function(error_history, parameters):
    k_p, k_i, k_d = parameters
    dE_dt = error_history[-1] - error_history[-2]
    E = error_history[-1]
    U = k_p * E + k_d * dE_dt + k_i * sum(error_history)
    return U


# Neural net controller
# TODO: Implement neural net controller function

# Bathtub constants
H_0 = 10
A = 10
c = 100
C = A / c
g = 9.81
T_bathtub = 0


# Bathtub input functions and variables
def bathtub_function(U, D, state):
    new_state = (U + D - C * np.sqrt(2 * g * state)) / A
    return new_state


# Cournot constants
p_max = 200
c_m = 1 / 20
T_cournot = 400


# Cournot function and variables
def cournot_function(U, D, state):
    q_1, q_2 = state
    new_state = (U + q_1) * (p_max - c_m - q_1 - q_2 - U - D)
    return new_state

# TODO: New problem constants...
