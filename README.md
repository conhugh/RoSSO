# RoSSO
RoSSO is a Python library for robotic surveillance strategy optimization. RoSSO utilizes JAX and Optax to provide a gradient-based optimization framework with a modular architecture. 

## Intro:
![rosso_graph_eg](https://github.com/conhugh/RoSSO/assets/95050521/ae130d2c-d7f8-4c6d-824f-d7befe3eb05a)
RoSSO is focused on a class of robotic surveillance problems with the following common structure:
 - One or more mobile robots patrolling an environment to deter bad actors
 - Environment modeled as a graph wherein: 
      - nodes correspond to locations of interest to bad actors
      - edges represent the paths that the robots can take between locations
      - edge lengths represent the robots' required travel times between locations
 - Robots and bad actors both take actions in discrete time
 - Robots' navigation of the environment is described by a Markov Chain (representing a "surveillance strategy")

 This structure allows for considerable variety in the details of the problem formulation. For example, explicit models for the capabilities of the robots and bad actors can be incorporated, patrol teams can be comprised of heterogeneous robot types, nodes can be given varying priority, etc. Recent literature also includes cases where no model for the bad actor is specified at all, and instead various heuristics are used to evaluate surveillance strategies for speed or unpredictability of coverage. 
 
 Note that in using Markov Chains for surveillance strategies, this structure does allow for deterministic patrolling of the graph. However, deterministic patrols are typically less effective than stochastic patrol strategies, as they are easier to exploit in most problem formulations. The superiority of stochastic strategies can arise either implicitly (through choice of heuristic) or explicitly (through modeling of bad actors capabilities). 
 
 As of ICRA 2024, RoSSO includes implementations of a few such problems, but the codebase is designed to be easily extended to study new ones. 

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
### problem_specs:
To support researchers in running repeatable, traceable, and organized computational studies at scale, RoSSO's architecture is designed such that all the information specifying a surveillance problem and a corresponding strategy optimization method is expected to be centralized. 

Any investigation involving one environment graph and one optimization approach can be fully specified by a single JSON file, which we refer to as a "problem spec". When running a computational study, a copy of the corresponding problem_spec JSON file will be saved together with the optimization results and any metrics which were tracked. This ensures that the parameters which led to each outcome can easily be determined retroactively.

Each JSON file is restricted to representing a single graph and a single optimization method in order to prevent the contents of each JSON file from becoming quite complex and unwieldy. Of course, you can modify the code to change this, but we do not recommend doing so. When sweeping hyperparameters, for example, it is recommended to programatically generate the necessary JSON files for each parameter combination. This may seem inefficient, but in practice, it turns out to be extremely helpful to associate a concise, isolated, and easily-readable problem spec with each set of results. 

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
metric_tracker.py defines the MetricTracker class, which provides infrastructure for handling typical "overhead" tasks involved in tracking any metric of interest throughout the iterative strategy optimization process. Using MetricTracker objects streamlines the process of defining new metrics and selecting metrics to track during a given computational study. This approach also allows for cleaner code in the main optimization loop (found in strat_opt.py). See the section "Adding a New Metric" below for additional details.

#### metric_definitions.py:
metric_definitions.py is a module which is used to define new functions that compute quantities of interest (i.e., "metrics") that can be tracked during the optimization process. These functions are given names via the keys in the "METRICS_REGISTRY". These names can then be listed within a problem specification JSON file to specify which metrics shall be tracked. 

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


