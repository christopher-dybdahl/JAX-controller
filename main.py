import constants
from controller import Controller
from plant import Plant
from consys import Consys

if __name__ == '__main__':
    classicPID = Controller(constants.classic_function, constants.classic_init_param)
    # TODO: Solve problems with neural net PID controller

    # Bathtub problem
    bathtub = Plant(constants.bathtub_function)
    consys_bathtub = Consys(bathtub, classicPID, constants.T_bathtub)
    optimal_U_bathtub, optimal_Y_bathtub, final_error_bathtub = consys_bathtub.simulate(constants.epochs, constants.timesteps)

    # Cournot problem
    cournot = Plant(constants.cournot_function)
    consys_cournot = Consys(cournot, classicPID, constants.T_cournot)
    optimal_U_cournot, optimal_Y_cournot, final_error_cournot = consys_bathtub.simulate(constants.epochs, constants.timesteps)
