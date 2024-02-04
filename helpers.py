import numpy as np


def noise(noise_range, noise_dim):
    lower_bound, upper_bound = noise_range
    noise = np.random.uniform(low=lower_bound, high=upper_bound, size=noise_dim)
    return noise