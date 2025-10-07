# JAX Controller

**IT3105 - Artificial Intelligence Programming - Project 1**

A modular Python implementation comparing classic PID controllers with neural network controllers for various control systems, built using JAX for automatic differentiation and gradient-based optimization.

## Overview

This project implements and compares two types of controllers across multiple plant models:
- **Classic PID Controller**: Traditional Proportional-Integral-Derivative controller
- **Neural Network Controller**: Multi-layer neural network with configurable architecture

The controllers are tested on three different plant models:
1. **Bathtub Model**: Water level control system with physics-based dynamics
2. **Battery Model**: Battery charge control system
3. **Cournot Model**: Economic production optimization (game theory)

## Key Features

- **JAX-based Implementation**: Leverages JAX for automatic differentiation and JIT compilation
- **Modular Architecture**: Clean separation between controllers, plants, and control systems
- **Configurable Parameters**: Centralized configuration for easy experimentation
- **Visualization**: Built-in plotting for MSE history and parameter evolution
- **Multiple Activation Functions**: Support for sigmoid, tanh, and ReLU in neural networks

## Project Structure

```
├── main files
│   ├── controller.py           # Abstract base controller class
│   ├── classic_controller.py   # PID controller implementation
│   ├── neural_net_controller.py # Neural network controller
│   ├── plant.py               # Abstract plant model
│   ├── consys.py              # Control system simulation engine
│   └── config.py              # Configuration and model definitions
├── run scripts
│   ├── bathtub_run.py         # Bathtub control simulation
│   ├── battery_run.py         # Battery control simulation
│   └── cournot_run.py         # Cournot game simulation
├── utilities
│   ├── helpers.py             # Utility functions (MSE, activation functions, noise)
│   └── test.py                # JAX functionality tests
└── project files
    ├── pyproject.toml         # Poetry dependency management
    ├── poetry.lock            # Locked dependencies
    └── README.md              # This file
```

## Requirements

- **Python**: ^3.11
- **JAX**: ^0.7.2
- **Additional**: numpy, matplotlib, scikit-learn (for neural network utilities)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd JAX-controller
   ```

2. **Install dependencies using Poetry:**
   ```bash
   poetry install
   ```

   Or using pip:
   ```bash
   pip install jax numpy matplotlib scikit-learn
   ```

## Usage

### Quick Start

Run any of the simulation scripts to see the controllers in action:

```bash
# Bathtub water level control
python bathtub_run.py

# Battery charge control  
python battery_run.py

# Cournot economic model
python cournot_run.py
```

### Configuration

All simulation parameters are centralized in `config.py`:

```python
# Training parameters
epochs = 1000
timesteps = 3  # Use 3 for cournot, 20 for others
learning_rate = 0.1

# Controller parameters
classic_init_param = [0.1, 0.1, 0.1]  # [Kp, Ki, Kd]
layers = [3, 5, 5, 1]  # Neural network architecture
activation_functions = [tanh, ReLU, sigmoid]
```

### Example Output

Each simulation produces:
- **MSE Learning Curves**: Shows convergence of both controllers
- **Parameter Evolution**: Tracks how controller parameters change during training
- **Performance Comparison**: Side-by-side comparison of classic PID vs neural network

## Plant Models

### 1. Bathtub Model
- **Physics**: Water level dynamics with inflow, outflow based on height
- **Control Goal**: Maintain target water level
- **Equation**: `h_new = h + (U + D - C√(2gh))/A`

### 2. Battery Model  
- **Physics**: Charge accumulation with current input
- **Control Goal**: Maintain target charge level
- **Equation**: `charge_new = charge + (U + I + D)/C`

### 3. Cournot Model
- **Economics**: Duopoly production game
- **Control Goal**: Maximize profit in competitive market
- **Equation**: Profit optimization under production constraints

## Architecture

### Controller Interface
```python
class Controller:
    def update_params(self, gradients)  # Parameter updates
    def update(self, parameters, error) # Control output computation
```

### Plant Interface
```python
class Plant:
    def update(self, U, D)  # State evolution with control U and disturbance D
    def get_state()         # Current plant state
```

### Control System
```python
class Consys:
    def simulate(epochs, timesteps, noise_range, U_init)  # Run simulation
    def print_mse_history(title)                          # Visualize learning
```

## Technical Details

- **Automatic Differentiation**: JAX automatically computes gradients for parameter updates
- **JIT Compilation**: Critical simulation loops are compiled for performance
- **Functional Programming**: Stateless functions enable efficient differentiation
- **Noise Handling**: Configurable disturbances test controller robustness

## Development

When adding new features:

1. **New Controllers**: Inherit from `Controller` base class
2. **New Plants**: Inherit from `Plant` base class  
3. **New Models**: Add configuration to `config.py`
4. **Dependencies**: Update using `poetry add <package>` or `pip freeze > requirements.txt`

## Academic Context

This project was developed for **IT3105 - Artificial Intelligence Programming** and demonstrates:
- Gradient-based optimization for control systems
- Comparison of traditional vs. AI-based control approaches
- JAX framework for scientific computing
- Modular software design for research applications

## License

Academic project - see course requirements for usage guidelines.