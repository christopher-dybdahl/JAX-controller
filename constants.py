import numpy as np
import jax.numpy as jnp

# Epoch and timesteps
epochs = 10
timesteps = 20


# Loss function
def MSE(model_function, X, w, y_true):
    y_pred = model_function(X, w)
    return jnp.mean((y_pred - y_true) ** 2)


# Controller functions and parameters
# Generic Model of a standard 3-parameter PID Controller
learning_rate = 0.001
classic_init_param = jnp.asarray([0.1, 0.01, 0.001])
noise_range = (-0.01, 0.01)


def classic_function(X, w):
    error_history = X
    k_p, k_i, k_d = w

    if len(X) <= 2:
        dE_dt = error_history[-1] - error_history[-2]
    else:
        dE_dt = error_history[-1]

    E = error_history[-1]
    sum_E = sum(error_history)

    # Function
    U = k_p * E + k_d * dE_dt + k_i * sum_E
    return U


# Neural net controller
# TODO: Implement neural net controller function

# Bathtub constants
H_0_bathtub = 10
A = 10
c = 100
C = A / c
g = 9.81
T_bathtub = H_0_bathtub
error_init = 4


# Bathtub input functions and variables
def bathtub_function(U, D, state):
    new_state = state + (U + D - C * np.sqrt(2 * g * state)) / A
    return new_state


# Cournot constants
H_0_cournot = 0
p_max = 200
c_m = 1 / 20
T_cournot = 400


# Cournot function and variables
def cournot_function(U, D, state):
    q_1, q_2 = state
    new_state = (U + q_1) * (p_max - c_m - q_1 - q_2 - U - D)
    return new_state

# TODO: New problem constants...
