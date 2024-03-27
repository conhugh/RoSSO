# RoSSO
RoSSO is a library for robotic surveillance strategy optimization. The strategies are represented by Markov chains. RoSSO utilizes JAX and Optax to provide a gradient-based optimization framework with a modular architecture. 

## Installation:
It is recommended to use a virtual environment when installing dependencies. If unfamiliar, see https://docs.python.org/3/library/venv.html.

If you only want to run computations locally and only on CPU, all required packages can be installed from the requirements.txt file after cloning the repo, as follows: 

      cd /path/to/your/RoboSurvStratOpt
      pip install -r requirements.txt

The main optimization-related dependencies are [Jax](https://jax.readthedocs.io/en/latest/index.html) and [Optax](https://optax.readthedocs.io/en/latest/). If you plan to use Google Cloud TPUs or Google Colab, or if you want to use the GPU in your graphics card to accelerate gradient computation on your local machine, you'll need to set up Jax separately from installing the other packages.  

Installation instructions can be found here: https://github.com/google/jax#installation. When enabling GPU-accelerated computation on your local machine, you will need to install the NVIDIA CUDA toolkit, cuDNN library, and appropriate driver for your graphics card (see links in the Jax installation instructions). You must have a CUDA-capable graphics card for this to work (see the list here: https://developer.nvidia.com/cuda-gpus). You will also need to sign up for the NVIDIA Developer Program to download and install cuDNN. 

For reference, on a Dell XPS 15 laptop with an Intel Core i7-10875H CPU (2.30 GHz) and a NVIDIA GeForce GTX 1650 Ti graphics card, gradients can be computed ~1000x faster when using the GPU (tested for environment graphs with ~10-1000 nodes).

Note that **if you have a Windows machine**, installing these dependencies will be easier if you first set up WSL2 and install Ubuntu (see tutorial here: https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview). As of January 2023, this is supposed to be possible on Windows 10, and Windows 11 is no longer required. Using WSL2 + Ubuntu makes the process simpler, both because Jax provides wheels for installation on Linux only (which you can use with WSL2) and because installing the CUDA toolkit is easier if you download and use the "WSL-Ubuntu" version.

## Quick Start Guide:

## Repo Organization:

### robosurvstratopt:
#### graph_gen.py: 
graph_gen.py provides methods for generating the adjacency and weight matrices corresponding to a variety of graph topologies including star graphs, complete graphs, grid graphs, etc. 

#### graph_comp.py:
graph_comp.py provides methods for analyzing, encoding, and decoding a given adjacency matrix. 

#### strat_comp.py: 
strat_comp.py contains methods defining a variety of objective functions for Markov chain optimization. Additionally, a few miscellaneous methods are provided at the top for purposes including generating random initial patrol strategies, pre-computing certain relevant quantities, and performing the parametrization that enforces a valid Markov chain transition matrix.
 
#### patrol_problem.py:
patrol_problem.py contains the definition of the PatrolProblem class. Notable methods contained within this class include initialize() which initializes various parameters based on the values provided in the .json file, compute_loss_and_gradient() which utilizes JAX's reverse-mode autodiff functionality to compute loss and gradient values for the various objective functions implemented in strat_comp.py, and cnvg_check() which keeps track of a moving average for determining convergence to the specified radius.

#### metric_tracker.py:


#### metric_definitions.py:


#### strat_opt.py: 
strat_opt.py is the main module that performs the gradient-based Markov chain optimization. run_test() takes a given PatrolProblem instance and optimizes each randomly initialized patrol strategy. run_optimizer() performs the first-order optimization of each patrol strategy using the Optax library. The step() function performs a single iteration of the desired optimization algorithm and is decorated with JAX's just-in-time compilation tag for improved performance. setup_optimizer() instantiates the appropriate Optax optimizer as specified in the .json file. 

#### strat_viz.py: 
strat_viz.py provides a host of useful methods for visualizing graphs, optimized patrol strategies, and the evolution of other metrics of interest during optimization. 
      
## Extending This Repo:

### Adding a new objective function:
1. Define the loss function in strat_comp.py. The first argument provided should always be Q, the arbitrary nxn matrix whose parametrization yields the Markov chain transition matrix P. Be sure to include the jit tag and specify the static arguments (see https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#python-control-flow-jit for more information).
2. Create a short string identifier for the new objective function and specify it in your .json file. Also, define within the .json file any new parameters that will be needed. 
3. Add a corresponding case to the if-elif structure in compute_loss_and_gradient() in patrol_problem.py. Also, ensure that any new parameters defined in your .json file are handled appropriately in initialize().

### Adding a new metric:


