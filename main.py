import matplotlib.pyplot as plt

import constants
from consys import Consys
from controller import Controller
from plant import Plant
import jax.numpy as jnp


if __name__ == '__main__':
    classicPID = Controller(constants.classic_function, constants.classic_init_param, constants.learning_rate)
    # TODO: Solve problems with neural net PID controller

    # Bathtub problem
    bathtub = Plant(constants.bathtub_function, constants.H_0_bathtub)
    consys_bathtub = Consys(bathtub, classicPID, constants.T_bathtub, constants.MSE)
    optimal_U_bathtub, optimal_Y_bathtub = consys_bathtub.simulate(constants.epochs,
                                                                   constants.timesteps,
                                                                   constants.noise_range,
                                                                   constants.error_init)
    consys_bathtub.print_mse_history()
    consys_bathtub.print_parameter_history()

    # Cournot problem
    cournot = Plant(constants.cournot_function, constants.H_0_cournot)
    consys_cournot = Consys(cournot, classicPID, constants.T_cournot, constants.MSE)
    optimal_U_cournot, optimal_Y_cournot = consys_bathtub.simulate(constants.epochs,
                                                                   constants.timesteps,
                                                                   constants.noise_range,
                                                                   constants.error_init)
    consys_bathtub.print_mse_history()
    consys_bathtub.print_parameter_history()
