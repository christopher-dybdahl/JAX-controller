import matplotlib.pyplot as plt

import constants
from consys import Consys
from controller import Controller
from plant import Plant


if __name__ == '__main__':
    classicPID = Controller(constants.classic_function, constants.classic_init_param, constants.learning_rate)
    # TODO: Solve problems with neural net PID controller

    # Bathtub problem
    # bathtub = Plant(constants.bathtub_function, constants.H_0_bathtub)
    # consys_bathtub = Consys(bathtub, classicPID, constants.T_bathtub, constants.MSE)
    # consys_bathtub.simulate(constants.epochs,
    #                         constants.timesteps,
    #                         constants.noise_range,
    #                         constants.U_bathtub_init)
    # consys_bathtub.print_mse_history()
    # consys_bathtub.print_parameter_history()

    # Cournot problem
    cournot = Plant(constants.cournot_function, constants.H_0_cournot)
    consys_cournot = Consys(cournot, classicPID, constants.T_cournot, constants.MSE, state_eval=constants.cournot_eval)
    consys_cournot.simulate(constants.epochs,
                            constants.timesteps,
                            constants.noise_range,
                            constants.U_cournot_init)
    consys_cournot.print_mse_history()
    consys_cournot.print_parameter_history()
