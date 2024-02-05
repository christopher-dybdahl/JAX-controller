from config import classicPID, neuralnetPID, epochs, timesteps, battery, noise_range_battery, U_init_battery, T_battery
from consys import Consys
from helpers import MSE

if __name__ == '__main__':
    # Initialize and run battery consys with classic PID controller
    consys_battery_classic = Consys(battery, classicPID, T_battery, MSE)
    consys_battery_classic.simulate(epochs, timesteps, noise_range_battery, U_init_battery)
    consys_battery_classic.print_mse_history("Classic PID Controller")
    consys_battery_classic.print_parameter_history("Classic PID Controller", legends=["Kp", "Kd", "Ki"])

    # Initialize and run bathtub consys with neural net PID controller
    consys_battery_NN = Consys(battery, neuralnetPID, T_battery, MSE)
    consys_battery_NN.simulate(epochs, timesteps, noise_range_battery, U_init_battery)
    consys_battery_NN.print_mse_history("Neural Network PID Controller")
