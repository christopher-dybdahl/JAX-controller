import numpy as np
import jax.numpy as jnp


def noise(noise_range, noise_dim):
    lower_bound, upper_bound = noise_range
    noise = np.random.uniform(low=lower_bound, high=upper_bound, size=noise_dim)
    return noise


def MSE(errors):
    return jnp.mean(errors ** 2)


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def tanh(x):
    return jnp.tanh(x)


def ReLU(x):
    return jnp.maximum(0, x)
