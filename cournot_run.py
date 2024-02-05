from config import classicPID, neuralnetPID, epochs, timesteps, cournot, noise_range_cournot, U_init_cournot, T_cournot, cournot_eval
from consys import Consys
from helpers import MSE

if __name__ == '__main__':
    # Initialize and run cournot consys with classic PID controller
    consys_cournot_classic = Consys(cournot, classicPID, T_cournot, MSE, state_eval=cournot_eval)
    consys_cournot_classic.simulate(epochs, timesteps, noise_range_cournot, U_init_cournot)
    consys_cournot_classic.print_mse_history("Classic PID Controller")
    consys_cournot_classic.print_parameter_history("Classic PID Controller", legends=["Kp", "Kd", "Ki"])

    # Initialize and run cournot consys with neural net PID controller
    consys_cournot_NN = Consys(cournot, neuralnetPID, T_cournot, MSE, state_eval=cournot_eval)
    consys_cournot_NN.simulate(epochs, timesteps, noise_range_cournot, U_init_cournot)
    consys_cournot_NN.print_mse_history("Neural Network PID Controller")
