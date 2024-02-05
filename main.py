import constants
from consys import Consys
from helpers import MSE
from neural_net_controller import Neural_net_controller
from pid_controller import Classic_controller
from plant import Plant

if __name__ == '__main__':
    # Initialize PID controllers
    classicPID = Classic_controller(constants.classic_init_param, constants.input_processor, constants.learning_rate)
    neuralnetPID = Neural_net_controller(constants.input_processor, constants.learning_rate, constants.layers,
                                         constants.activation_functions)

    # Initialize bathtub problem
    bathtub = Plant(constants.bathtub_function, constants.H_0_bathtub)

    # Initialize bathtub consys with classic PID controller and run
    consys_bathtub_classic = Consys(bathtub, classicPID, constants.T_bathtub, MSE)
    consys_bathtub_classic.simulate(constants.epochs,
                                    constants.timesteps,
                                    constants.noise_range,
                                    constants.U_bathtub_init)
    consys_bathtub_classic.print_mse_history("consys_bathtub_classic - Bathtub")
    consys_bathtub_classic.print_parameter_history("consys_bathtub_classic - Bathtub")

    # Initialize bathtub consys with neural net PID controller and run
    consys_bathtub_NN = Consys(bathtub, neuralnetPID, constants.T_bathtub, MSE)
    consys_bathtub_NN.simulate(constants.epochs,
                               constants.timesteps,
                               constants.noise_range,
                               constants.U_bathtub_init)
    consys_bathtub_NN.print_mse_history("consys_bathtub_NN - Bathtub")

    # Initialize cournot problem
    cournot = Plant(constants.cournot_function, constants.H_0_cournot)

    # Initialize cournot consys with classic PID controller and run
    consys_cournot_classic = Consys(cournot, classicPID, constants.T_cournot, MSE,
                                    state_eval=constants.cournot_eval)
    consys_cournot_classic.simulate(constants.epochs,
                                    constants.timesteps,
                                    constants.noise_range,
                                    constants.U_cournot_init)
    consys_cournot_classic.print_mse_history("consys_cournot_classic - Cournot")
    consys_cournot_classic.print_parameter_history("consys_cournot_classic - Cournot")

    # Initialize cournot consys with neural net PID controller and run
    consys_cournot_NN = Consys(cournot, neuralnetPID, constants.T_cournot, MSE,
                               state_eval=constants.cournot_eval)
    consys_cournot_NN.simulate(constants.epochs,
                               constants.timesteps,
                               constants.noise_range,
                               constants.U_cournot_init)
    consys_cournot_NN.print_mse_history("consys_cournot_NN - Cournot")

    # Initialize battery problem
    battery = Plant(constants.battery_function, constants.H_0_battery)

    # Initialize battery consys with classic PID controller and run
    consys_battery_classic = Consys(battery, classicPID, constants.T_battery, MSE)
    consys_battery_classic.simulate(constants.epochs,
                                    constants.timesteps,
                                    constants.noise_range_battery,
                                    constants.U_battery_init)
    consys_battery_classic.print_mse_history("consys_battery_classic - Battery")
    consys_battery_classic.print_parameter_history("consys_battery_classic - Battery")

    # Initialize bathtub consys with neural net PID controller and run
    consys_battery_NN = Consys(battery, neuralnetPID, constants.T_battery, MSE)
    consys_battery_NN.simulate(constants.epochs,
                               constants.timesteps,
                               constants.noise_range_battery,
                               constants.U_battery_init)
    consys_battery_NN.print_mse_history("consys_battery_NN - Battery")
