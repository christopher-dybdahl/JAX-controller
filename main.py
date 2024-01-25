import matplotlib.pyplot as plt

import constants
from consys import Consys
from controller import Controller
from plant import Plant

if __name__ == '__main__':
    classicPID = Controller(constants.classic_function, constants.classic_init_param)
    # TODO: Solve problems with neural net PID controller

    # Bathtub problem
    bathtub = Plant(constants.bathtub_function)
    consys_bathtub = Consys(bathtub, classicPID, constants.T_bathtub)
    optimal_U_bathtub, optimal_Y_bathtub, mse_history, params_history = consys_bathtub.simulate(constants.epochs, constants.timesteps)

    plt.plot(mse_history)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Learning Progression')
    plt.legend()
    plt.show()

    plt.plot(params_history)
    plt.xlabel('Epoch')
    plt.ylabel('Y')
    plt.title('Control Parameters')
    plt.show()

    # Cournot problem
    cournot = Plant(constants.cournot_function)
    consys_cournot = Consys(cournot, classicPID, constants.T_cournot)
    optimal_U_cournot, optimal_Y_cournot, final_error_cournot = consys_bathtub.simulate(constants.epochs, constants.timesteps)

    # TODO: Check if better visuals method

    plt.plot(mse_history)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Learning Progression')
    plt.legend()
    plt.show()

    plt.plot(params_history)
    plt.xlabel('Epoch')
    plt.ylabel('Y')
    plt.title('Control Parameters')
    plt.show()
