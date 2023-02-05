# RoboSurvStratOpt
Repository for research code related to robotic surveillance problems. 

## Installing Dependencies:
The main optimization-related packages are Jax and Optax, more info can be found here:
 - Jax (https://jax.readthedocs.io/en/latest/index.html) 
 - Optax (https://optax.readthedocs.io/en/latest/) 

It is recommended to use a virtual environment when installing dependencies. If unfamiliar, see https://docs.python.org/3/library/venv.html.

If you only plan to run computations locally and only on CPU, all required packages can be installed simply by running: 
      pip install -r requirements.txt

If you plan to run computations with Google Cloud TPUs or in Google Colab, or if you want to use the GPU in your graphics card to accelerate gradient computation on your local machine, you'll need to set up Jax separately from installing the other packages.

Installation instructions can be found here: https://github.com/google/jax#installation. When enabling GPU-accelerated computation on your local machine, the most involved part will be installing the NVIDIA CUDA toolkit, cuDNN library, and appropriate driver for your graphics card. You will need to have a CUDA-capable graphics card (see the list here: https://developer.nvidia.com/cuda-gpus). You will also need to sign up for the NVIDIA Developer Program to download and install cuDNN. 

Note that if you have a Windows machine, installing these dependencies will be easier if you set up WSL2 and install Ubuntu (see tutorial here: https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview). As of January 2023, you are supposed to be able to do this on Windows 10 and no longer need Windows 11. Using WSL2 + Ubuntu makes the process simpler, both because Jax provides wheels for installation on Linux only (which you can use with WSL2) and because installing the CUDA toolkit is easier if you download and use the "WSL-Ubuntu" version.

## Comments on Organization:
Overall layout of the codebase is as follows:
#### GraphGen.py: 
      - generating, storing, and loading environment graphs, 
      - extracting simple information about those graphs (e.g., graph diameter, set of leaf nodes, etc.) 
#### StratCompJax.py:
      - initializing surveillance strategies 
      - computing capture probabilities 
      - computing various gradients (including the MCP gradient) 
      - projecting updated strategies onto the constraint set (no longer used) 
      - parametrizing the constraints (see Docs/'Notes on Parametrization vs Projection.pdf')
      - accounting for symmetries in comparing grid graph strategies 
#### StratOptOptax.py: 
      - testing the performance of various gradient-based optimization methods 
      - loading parameters/settings for desired optimization method 
      - running gradient-based optimization for provided graphs and initial strategies 
      - checking for strategy convergence during the optimization process 
      - tracking and saving various metrics used to visualize performance of desired optimization method 
#### StratViz.py: 
      - visualizing surveillance strategies by drawing and labeling graphs 
      - visualizing metrics tracked during optimization processes 
      - visualizing statistics about the results of strategy optimization processes 
#### TestSpec.py:
      - defines a "TestSpec", which stores information about computational studies including: 
          - environment graphs, with encodings that speed up adjacency matrix generation 
          - number of random initial strategies to use for each graph 
          - settings to use for the optimization process 
      - the TestSpec class enables saving and validates TestSpecs before running optimization processes 
#### Testing.py: 
      - Connor's scratch paper -- recently used for studying tree graph properties
#### ModifyTestSpec.py:
      - Not currently used
#### StratOptColab.ipynb:
      - Was used briefly to try running optimizations on GPU and TPU in Google Colab, not currently used
