# Learning the Optimal Power Flow: Environment Design Matters
This is the accompanying repository to the publication "[Learning the Optimal Power Flow: Environment Design Matters](https://arxiv.org/abs/2403.17831)".

Note that most source code implemented for this paper can be found in 
https://gitlab.com/thomaswolgast/drl (RL algorithms) and 
https://github.com/Digitalized-Energy-Systems/opfgym 
(RL environments, warning: repository re-named and moved from GitLab to GitHub), 
which are work-in-progress and therefore continued in different repositories. 

# Installation
All experiments were performed with python 3.8. In an virtualenv, run `pip install -r requirements.txt` to install all dependencies in the right version at publication time (not the most recent). 

Note: [torch](https://pytorch.org/get-started/locally/) sometimes makes problems and needs to be installed manually before performing the previous step.

# Repository structure
- `run.sh`: A list of the commands performed to reproduce all experiments done for this publication. **Should not be run all at once!** Overall computation time will be multiple weeks. Use this file to copy-paste single commands from. Will automatically create a `data/` folder with the result files. 
- `LICENSE`: The license used for this work (MIT). 
- `requirements.txt`: Reference to the two previously mentioned repositories for simple installation. 
- `src/`: The source code to aggregate the results and create the figures of this exact publication. The source code for running the experiments is in external repositories.
- `compact_results/`: Collection of JSON files with compact representations of the results with different metrics. (created with `create_results.py` and used by `boxplots.py`)
- `results/`: **[Optional folder that can be downloaded from Zenodo](https://zenodo.org/records/13284446)**. Contains model weights, hyperparameter information, and agent performances in the course of training (e.g. MAPE every 30k steps). The model weights are required to run `create_results.py`. The training performances are required to plot performance during training, if required. Download this folder from Zenodo and place it in this repository to work with the data. 

Attention: Sometimes, the VoltageControl environment is called QMarketEnv because of a name change during development. Both names are synonymous in the context of this work. 

# Contact
For questions or feedback, contact the first author Thomas Wolgast (thomas.wolgast@uol.de). 
