# Learning the Optimal Power Flow: Environment Design Matters
This is the accompanying repository to the publication "Learning the Optimal Power Flow: Environment Design Matters" (TODO: Add URL to paper).
Note that most source code implemented for this paper can be found in https://gitlab.com/thomaswolgast/drl (RL algorithms) and https://gitlab.com/thomaswolgast/mlopf (RL environments), which are work-in-progress and therefore continued in different repositories. 

# Installation
All experiments were performed with python 3.8. In an virtualenv, run `pip install git+https://gitlab.com/thomaswolgast/mlopf.git@f038cd591158a909236b06813232505e370b95ba` and `pip install git+https://gitlab.com/thomaswolgast/drl.git@aa1205a95393c40db08697be4c5ec73d83f81f4b` to install all dependencies in the right version at publication (not the most recent). 

Note: [torch](https://pytorch.org/get-started/locally/) sometimes makes problems and needs to be installed manually before performing the previous steps.

# Repository structure
- `src/`: The source code to aggregate the results and create the figures of this exact publication. All the source code for running the experiments is in external repositories.
- `compact_results/`: Collection of JSON files with compact representations of the results with different metrics. (created with `create_results.py`)
- `results/`: The comprehensive results of all experiments performed for this publication. (Note: The environment `VoltageControlEnv` is called `QMarketEnv` here because of a name change during development)
- `run.sh`: A list of the commands performed to start all experiments done for this publication. Should not be run all at once! Overall computation time will be multiple weeks. Use this file to copy-paste single commands from.
- `LICENSE`: The license used for this work (MIT). 

# Contact
For questions or feedback contact the first author Thomas Wolgast (thomas.wolgast@uol.de). 
