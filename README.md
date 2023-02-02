# RoboSurvStratOpt
Repository for research code related to robotic surveillance problems. 

## Dependencies:
 - Jax (https://jax.readthedocs.io/en/latest/index.html) 
    To optimize strategies on GPU, will need to follow the install instructions here: https://github.com/google/jax#pip-installation-gpu-cuda 
 - Optax (https://optax.readthedocs.io/en/latest/) 
 - numpy, matplotlib, networkx, pygraphviz ... will add a requirements.txt file soon to make the environment easier to replicate. 


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
