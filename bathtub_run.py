from config import classicPID, neuralnetPID, epochs, timesteps, bathtub, noise_range_bathtub, U_init_bathtub, T_bathtub
from consys import Consys
from helpers import MSE

if __name__ == '__main__':
    # Initialize and run bathtub consys with classic PID controller
    consys_bathtub_classic = Consys(bathtub, classicPID, T_bathtub, MSE)
    consys_bathtub_classic.simulate(epochs, timesteps, noise_range_bathtub, U_init_bathtub)
    consys_bathtub_classic.print_mse_history("Classic PID Controller")
    consys_bathtub_classic.print_parameter_history("Classic PID Controller", legends=["Kp", "Kd", "Ki"])

    # Initialize and run bathtub consys with neural net PID controller
    consys_bathtub_NN = Consys(bathtub, neuralnetPID, T_bathtub, MSE)
    consys_bathtub_NN.simulate(epochs, timesteps, noise_range_bathtub, U_init_bathtub)
    consys_bathtub_NN.print_mse_history("Neural Network PID Controller")
