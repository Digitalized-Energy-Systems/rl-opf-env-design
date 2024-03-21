# Publication: Learning the Optimal Power Flow: Environment Design Matters
This is the accompanying repository to the publication "Learning the Optimal Power Flow: Environment Design Matters" (TODO: Add link).
However, note that most source implemented for this paper can be found in TODO (RL code) and TODO (RL environments), which are work-in-progress and therefore continued in different repositories. 

# Repository Structure
- `src/`: The source code to aggregate the results and create the figures of this exact publication.
- `compact_results/`: Collection of JSON files with compact representations of the results with different metrics.
- `results/`: The comprehensive results of all experiments performed for this publication. 
- `run.sh`: A list of the commands performed to start all experiments done for this publication. Should not be run all at once! Overall computation time will be multiple weeks.
- `requirements.txt`: A list of all packages used for the experiments, including the RL and OPF-environment frameworks developed for this research.  

# Installation
Run `pip install -r requirements` to install all dependencies in the exact version used for the experiments in the publication. 

# Contact
For questions or feedback contact the first author Thomas Wolgast (thomas.wolgast@uol.de). 