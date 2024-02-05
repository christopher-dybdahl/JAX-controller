import jax.numpy as jnp
from helpers import sigmoid, tanh, ReLU

# Epoch and timesteps
epochs = 500
timesteps = 20

# Controller functions and parameters
learning_rate = 0.5
noise_range = (-0.01, 0.01)


def input_processor(X):
    error_history = X

    if len(X) <= 2:
        dE_dt = 0
    else:
        dE_dt = error_history[-1] - error_history[-2]

    E = error_history[-1]
    sum_E = jnp.sum(error_history)

    return jnp.array([E, dE_dt, sum_E])


# Classic controller parameters
classic_init_param = jnp.asarray([0.1, 0.1, 0.1])

# Neural net controller parameters
activation_functions = [sigmoid, tanh, ReLU]
layers = [3, 5, 8, 1]

# Bathtub parameters
H_0_bathtub = 10
U_bathtub_init = 1
T_bathtub = H_0_bathtub
A = 10
c = 100
C = A / c
g = 9.81


# Bathtub input functions
def bathtub_function(U, D, height):
    height_new = height + (U + D - C * jnp.sqrt(2 * g * height)) / A
    return height_new


# Cournot parameters
H_0_cournot = 0.1, 0.1
U_cournot_init = 0
T_cournot = 1
p_max = 2
c_m = 0.1
q_1_lower = 0
q_1_upper = 1
q_2_lower = 0
q_2_upper = 1


# Cournot functions
def cournot_function(U, D, state):
    q_1, q_2 = state

    q_1_new = U + q_1
    q_2_new = D + q_2

    # Constraints
    q_1_new = jnp.minimum(q_1_new, q_1_upper)
    q_1_new = jnp.maximum(q_1_new, q_1_lower)

    q_2_new = jnp.minimum(q_2_new, q_2_upper)
    q_2_new = jnp.maximum(q_2_new, q_2_lower)

    return q_1_new, q_2_new


def cournot_eval(Y):
    q_1, q_2 = Y
    return profit_function(q_1, q_2)


def profit_function(q_1, q_2):
    q = q_1 + q_2
    p = p_max - q
    P_1 = q_1 * (p - c_m)
    return P_1


# Battery parameters
noise_range_battery = (-1, 1)
H_0_battery = 0.6
U_battery_init = 0
T_battery = 0.8
I = 10
C_battery = 100


# Battery functions
def battery_function(U, D, charge):
    charge_new = charge + (U + I + D) / C_battery
    return charge_new
