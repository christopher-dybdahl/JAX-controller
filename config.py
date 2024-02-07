import jax.numpy as jnp

from classic_controller import Classic_controller
from helpers import sigmoid, tanh, ReLU
from neural_net_controller import Neural_net_controller
from plant import Plant

# General functions and parameters
epochs = 1000
timesteps = 20 # Use 5 for cournot, 20 else
learning_rate = 0.1


def input_processor(X):
    error_history = X

    if len(X) <= 2:
        dE_dt = 0
    else:
        dE_dt = error_history[-1] - error_history[-2]

    E = error_history[-1]
    sum_E = jnp.sum(error_history)

    return jnp.array([E, dE_dt, sum_E])


# Classic controller
classic_init_param = jnp.asarray([0.5, 0.5, 0.5]) # Use [0.5, 0.5, 0.5] for bathtub
classicPID = Classic_controller(classic_init_param, input_processor, learning_rate)

# Neural network controller
parameter_range = (0, 0.2)
activation_functions = [sigmoid, tanh, ReLU]
layers = [3, 5, 8, 1]
neuralnetPID = Neural_net_controller(input_processor, learning_rate, layers, parameter_range, activation_functions)

# Bathtub model
noise_range_bathtub = (-0.1, 0.1)
H_0_bathtub = 10
U_init_bathtub = 1
T_bathtub = H_0_bathtub
A = 10
c = 100
C = A / c
g = 9.81


def bathtub_function(U, D, height):
    height_new = height + (U + D - C * jnp.sqrt(2 * g * height)) / A
    return height_new


bathtub = Plant(bathtub_function, H_0_bathtub)

# Cournot model
noise_range_cournot = (-0.01, 0.01)
H_0_cournot = 0.2, 0.2
U_init_cournot = 0
T_cournot = 1.5
p_max = 3
c_m = 0.1
q_1_lower = 0
q_1_upper = 1
q_2_lower = 0
q_2_upper = 1


def cournot_function(U, D, productions):
    q_1, q_2 = productions

    q_1_new = U + q_1
    q_2_new = D + q_2

    # Constraints
    q_1_new = jnp.minimum(q_1_new, q_1_upper)
    q_1_new = jnp.maximum(q_1_new, q_1_lower)

    q_2_new = jnp.minimum(q_2_new, q_2_upper)
    q_2_new = jnp.maximum(q_2_new, q_2_lower)

    return q_1_new, q_2_new


def profit_function(q_1, q_2):
    q = q_1 + q_2
    p = p_max - q
    P_1 = q_1 * (p - c_m)
    return P_1


def cournot_eval(Y):
    q_1, q_2 = Y
    return profit_function(q_1, q_2)


cournot = Plant(cournot_function, H_0_cournot)

# Battery model
noise_range_battery = (-0.1, 0.1)
H_0_battery = 0.7
U_init_battery = 0
T_battery = 0.8
I = 10
C_battery = 100


# Battery functions
def battery_function(U, D, charge):
    charge_new = charge + (U + I + D) / C_battery
    return charge_new


battery = Plant(battery_function, H_0_battery)
