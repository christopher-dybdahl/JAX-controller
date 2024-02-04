import numpy as np
import jax.numpy as jnp

# Epoch and timesteps
epochs = 500
timesteps = 20


# Loss function
def MSE(errors):
    return jnp.mean(errors ** 2)


# Controller functions and parameters
# Generic Model of a standard 3-parameter PID Controller
learning_rate = 0.5
classic_init_param = jnp.asarray([0.1, 0.1, 0.1])
noise_range = (-0.01, 0.01)


def classic_function(X, parameters):
    error_history = X
    k_p, k_i, k_d = parameters

    if len(X) <= 2:
        dE_dt = 0
    else:
        dE_dt = error_history[-1] - error_history[-2]

    E = error_history[-1]
    sum_E = jnp.sum(error_history)

    # Function
    U = k_p * E + k_d * dE_dt + k_i * sum_E
    return U


# Neural net controller
def neural_net_function(X, layers=[5, 10, 5]):  # Activation?
    error_history = X

    if len(X) <= 2:
        dE_dt = 0
    else:
        dE_dt = error_history[-1] - error_history[-2]

    E = error_history[-1]
    sum_E = jnp.sum(error_history)

    sender = layers[0]
    params = []
    for receiver in layers[1:]:
        weights = np.random.uniform(-.1, .1, (sender, receiver))
        biases = np.random.uniform(-.1, .1, (1, receiver))
        sender = receiver
        params.append([weights, biases])
    return U

# Bathtub constants
H_0_bathtub = 10
A = 10
c = 100
C = A / c
g = 9.81
T_bathtub = H_0_bathtub
U_bathtub_init = 1


# Bathtub input functions and variables
def bathtub_function(U, D, height):
    height_new = height + (U + D - C * jnp.sqrt(2 * g * height)) / A
    return height_new


# Cournot constants
H_0_cournot = 0.1, 0.1
p_max = 2
c_m = 0.1
T_cournot = 1
U_cournot_init = 0
q_1_lower = 0
q_1_upper = 1
q_2_lower = 0
q_2_upper = 1


# Cournot function and variables
def cournot_function(U, D, state):
    q_1, q_2 = state

    q_1_new = U + q_1
    q_2_new = D + q_2

    # Constraints
    q_1_new = jnp.minimum(q_1_new, q_1_upper)
    q_1_new = jnp.maximum(q_1_new, q_1_lower)

    q_2_new = jnp.minimum(q_2_new, q_2_upper)
    q_2_new = jnp.maximum(q_2_new, q_2_lower)

    print(f"q_1: {q_1_new}, q_2: {q_2_new}")

    return q_1_new, q_2_new


def cournot_eval(Y):
    q_1, q_2 = Y
    return profit_function(q_1, q_2)


def profit_function(q_1, q_2):
    q = q_1 + q_2
    p = p_max - q
    P_1 = q_1 * (p - c_m)
    # print(f"P_1: {P_1}")
    return P_1

# TODO: New problem constants...
