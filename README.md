# RoboSurvStratOpt
Repository for research code related to robotic surveillance problems. 

## Dependencies:
 - Jax (https://jax.readthedocs.io/en/latest/index.html) \n
    To optimize strategies on GPU, will need to follow the install instructions here: https://github.com/google/jax#pip-installation-gpu-cuda \n
 - Optax (https://optax.readthedocs.io/en/latest/) \n
 - numpy, matplotlib, networkx, pygraphviz ... will add a requirements.txt file soon to make the environment easier to replicate. 


## Comments on Organization:
Overall layout of the codebase is as follows:\n
#### GraphGen.py: 
      - generating, storing, and loading environment graphs, \n
      - extracting simple information about those graphs (e.g., graph diameter, set of leaf nodes, etc.) 
#### StratCompJax.py:
      - initializing surveillance strategies \n
      - computing capture probabilities \n
      - computing various gradients (including the MCP gradient) \n
      - projecting updated strategies onto the constraint set (no longer used) \n
      - parametrizing the constraints (see Docs/'Notes on Parametrization vs Projection.pdf')\n
      - accounting for symmetries in comparing grid graph strategies \n\n
#### StratOptOptax.py: 
      - testing the performance of various gradient-based optimization methods \n
      - loading parameters/settings for desired optimization method \n
      - running gradient-based optimization for provided graphs and initial strategies \n
      - checking for strategy convergence during the optimization process \n
      - tracking and saving various metrics used to visualize performance of desired optimization method 
#### StratViz.py: 
      - visualizing surveillance strategies by drawing and labeling graphs \n
      - visualizing metrics tracked during optimization processes \n
      - visualizing statistics about the results of strategy optimization processes \n\n
#### TestSpec.py:
      - defines a "TestSpec", which stores information about computational studies including: \n
          - environment graphs, with encodings that speed up adjacency matrix generation \n
          - number of random initial strategies to use for each graph \n
          - settings to use for the optimization process 
      - the TestSpec class enables saving and validates TestSpecs before running optimization processes \n\n
#### Testing.py: 
      - Connor's scratch paper -- recently used for studying tree graph properties
#### ModifyTestSpec.py:
      - Not currently used
#### StratOptColab.ipynb:
      - Was used briefly to try running optimizations on GPU and TPU in Google Colab, not currently used
