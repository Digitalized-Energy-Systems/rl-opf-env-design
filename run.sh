<<Introduction
    Run these commands to repeat the experiments in the publication. 
    Note: We often use the "Ddpg1Step" algorithm, which allows for faster training in 1-step environments by omitting some irrelevant steps, without changing the results. 
Introduction

# Run VoltageControl experiments (here QMarketEnv)
## Baseline
python src/main.py --agent "Ddpg1Step" --enviro "mlopf.envs.thesis_envs:QMarketEnv" --store --steps 1000000 --test-ste 200 --test-inter 30000 --path "data/voltage_control/base/" --num 10 --hyper "{'batch_size': 1024, 'memory_size': 1000000, 'actor_fc_dims': (256, 256, 256), 'critic_fc_dims': (256, 256, 256)}" --env-hyper "{}"

## Training Data
python src/main.py --agent "Ddpg1Step" --enviro "mlopf.envs.thesis_envs:QMarketEnv" --store --steps 1000000 --test-ste 200 --test-inter 30000 --path "data/voltage_control/uniform_data/" --num 10 --hyper "{'batch_size': 1024, 'memory_size': 1000000, 'actor_fc_dims': (256, 256, 256), 'critic_fc_dims': (256, 256, 256)}" --env-hyper "{'train_data': 'full_uniform'}"
python src/main.py --agent "Ddpg1Step" --enviro "mlopf.envs.thesis_envs:QMarketEnv" --store --steps 1000000 --test-ste 200 --test-inter 30000 --path "data/voltage_control/normal_data/" --num 10 --hyper "{'batch_size': 1024, 'memory_size': 1000000, 'actor_fc_dims': (256, 256, 256), 'critic_fc_dims': (256, 256, 256)}" --env-hyper "{'train_data': 'normal_around_mean', 'sampling_kwargs': {'std': 0.3}}"

## Observation Space
python src/main.py --agent "Ddpg1Step" --enviro "mlopf.envs.thesis_envs:QMarketEnv" --store --steps 1000000 --test-ste 200 --test-inter 30000 --path "data/voltage_control/res_obs/" --num 10 --hyper "{'batch_size': 1024, 'memory_size': 1000000, 'actor_fc_dims': (256, 256, 256), 'critic_fc_dims': (256, 256, 256)}" --env-hyper "{'add_res_obs': True}"
python src/main.py --agent "Ddpg1Step" --enviro "mlopf.envs.thesis_envs:QMarketEnv" --store --steps 1000000 --test-ste 200 --test-inter 30000 --path "data/voltage_control/act_obs/" --num 10 --hyper "{'batch_size': 1024, 'memory_size': 1000000, 'actor_fc_dims': (256, 256, 256), 'critic_fc_dims': (256, 256, 256)}" --env-hyper "{'add_res_obs': True, 'add_act_obs': True}"

## Episode Definition
python src/main.py --agent "Ddpg" --enviro "mlopf.envs.thesis_envs:QMarketEnv" --store --steps 1000000 --test-ste 200 --test-inter 30000 --path "data/voltage_control/nstep_05/" --num 10 --hyper "{'batch_size': 1024, 'memory_size': 1000000, 'actor_fc_dims': (256, 256, 256), 'critic_fc_dims': (256, 256, 256), 'gamma': 0.5}" --env-hyper "{'add_res_obs': True, 'add_act_obs': True, 'steps_per_episode': 5}"
python src/main.py --agent "Ddpg" --enviro "mlopf.envs.thesis_envs:QMarketEnv" --store --steps 1000000 --test-ste 200 --test-inter 30000 --path "data/voltage_control/nstep_09/" --num 10 --hyper "{'batch_size': 1024, 'memory_size': 1000000, 'actor_fc_dims': (256, 256, 256), 'critic_fc_dims': (256, 256, 256), 'gamma': 0.9}" --env-hyper "{'add_res_obs': True, 'add_act_obs': True, 'steps_per_episode': 5}"

## Reward Function
python src/main.py --agent "Ddpg1Step" --enviro "mlopf.envs.thesis_envs:QMarketEnv" --store --steps 1000000 --test-ste 200 --test-inter 30000 --path "data/voltage_control/sum_10x/" --num 10 --hyper "{'batch_size': 1024, 'memory_size': 1000000, 'actor_fc_dims': (256, 256, 256), 'critic_fc_dims': (256, 256, 256)}" --env-hyper "{'reward_function': 'summation', 'ext_grid_pen_kwargs': {'linear_penalty': 5000}}"
python src/main.py --agent "Ddpg1Step" --enviro "mlopf.envs.thesis_envs:QMarketEnv" --store --steps 1000000 --test-ste 200 --test-inter 30000 --path "data/voltage_control/replace_min/" --num 10 --hyper "{'batch_size': 1024, 'memory_size': 1000000, 'actor_fc_dims': (256, 256, 256), 'critic_fc_dims': (256, 256, 256)}" --env-hyper "{'reward_function': 'replacement'}"
python src/main.py --agent "Ddpg1Step" --enviro "mlopf.envs.thesis_envs:QMarketEnv" --store --steps 1000000 --test-ste 200 --test-inter 30000 --path "data/voltage_control/replace_mean/" --num 10 --hyper "{'batch_size': 1024, 'memory_size': 1000000, 'actor_fc_dims': (256, 256, 256), 'critic_fc_dims': (256, 256, 256)}" --env-hyper "{'reward_function': 'replacementA'}"


# Perform EcoDispatch experiments
## Baseline
python src/main.py --agent "Ddpg1Step" --enviro "mlopf.envs.thesis_envs:EcoDispatchEnv" --store --steps 2000000 --test-ste 200 --test-inter 50000 --path "data/final_experiments/eco_dispatch/base/" --num 10 --hyper "{'batch_size': 1024, 'memory_size': 1000000, 'actor_fc_dims': (256, 256, 256, 256, 256, 256), 'critic_fc_dims': (256, 256, 256, 256, 256, 256)}" --env-hyper "{}"

## Training Data
python src/main.py --agent "Ddpg1Step" --enviro "mlopf.envs.thesis_envs:EcoDispatchEnv" --store --steps 2000000 --test-ste 200 --test-inter 50000 --path "data/final_experiments/eco_dispatch/uniform_data/" --num 10 --hyper "{'batch_size': 1024, 'memory_size': 1000000, 'actor_fc_dims': (256, 256, 256, 256, 256, 256), 'critic_fc_dims': (256, 256, 256, 256, 256, 256)}" --env-hyper "{'train_data': 'full_uniform'}"
python src/main.py --agent "Ddpg1Step" --enviro "mlopf.envs.thesis_envs:EcoDispatchEnv" --store --steps 2000000 --test-ste 200 --test-inter 50000 --path "data/final_experiments/eco_dispatch/normal_data/" --num 10 --hyper "{'batch_size': 1024, 'memory_size': 1000000, 'actor_fc_dims': (256, 256, 256, 256, 256, 256), 'critic_fc_dims': (256, 256, 256, 256, 256, 256)}" --env-hyper "{'train_data': 'normal_around_mean', 'sampling_kwargs': {'std': 0.3}}"

## Observation Space
python src/main.py --agent "Ddpg1Step" --enviro "mlopf.envs.thesis_envs:EcoDispatchEnv" --store --steps 2000000 --test-ste 200 --test-inter 50000 --path "data/final_experiments/eco_dispatch/res_obs/" --num 10 --hyper "{'batch_size': 1024, 'memory_size': 1000000, 'actor_fc_dims': (256, 256, 256, 256, 256, 256), 'critic_fc_dims': (256, 256, 256, 256, 256, 256)}" --env-hyper "{'add_res_obs': True}"
python src/main.py --agent "Ddpg1Step" --enviro "mlopf.envs.thesis_envs:EcoDispatchEnv" --store --steps 2000000 --test-ste 200 --test-inter 50000 --path "data/final_experiments/eco_dispatch/act_obs/" --num 10 --hyper "{'batch_size': 1024, 'memory_size': 1000000, 'actor_fc_dims': (256, 256, 256, 256, 256, 256), 'critic_fc_dims': (256, 256, 256, 256, 256, 256)}" --env-hyper "{'add_res_obs': True, 'add_act_obs': True}"

## Episode Definition
python src/main.py --agent "Ddpg" --enviro "mlopf.envs.thesis_envs:EcoDispatchEnv" --store --steps 2000000 --test-ste 200 --test-inter 50000 --path "data/final_experiments/eco_dispatch/nstep_05/" --num 10 --hyper "{'batch_size': 1024, 'memory_size': 1000000, 'actor_fc_dims': (256, 256, 256, 256, 256, 256), 'critic_fc_dims': (256, 256, 256, 256, 256, 256)}" --env-hyper "{'add_res_obs': True, 'add_act_obs': True, 'steps_per_episode': 5}"
python src/main.py --agent "Ddpg" --enviro "mlopf.envs.thesis_envs:EcoDispatchEnv" --store --steps 2000000 --test-ste 200 --test-inter 50000 --path "data/final_experiments/eco_dispatch/nstep_09/" --num 10 --hyper "{'batch_size': 1024, 'memory_size': 1000000, 'actor_fc_dims': (256, 256, 256, 256, 256, 256), 'critic_fc_dims': (256, 256, 256, 256, 256, 256)}" --env-hyper "{'add_res_obs': True, 'add_act_obs': True, 'steps_per_episode': 5}"

## Reward Function
python src/main.py --agent "Ddpg1Step" --enviro "mlopf.envs.thesis_envs:EcoDispatchEnv" --store --steps 2000000 --test-ste 200 --test-inter 50000 --path "data/final_experiments/eco_dispatch/sum_10x/" --num 10 --hyper "{'batch_size': 1024, 'memory_size': 1000000, 'actor_fc_dims': (256, 256, 256, 256, 256, 256), 'critic_fc_dims': (256, 256, 256, 256, 256, 256)}" --env-hyper "{'reward_function': 'summation', 'ext_grid_pen_kwargs': {'linear_penalty': 100000}}"
python src/main.py --agent "Ddpg1Step" --enviro "mlopf.envs.thesis_envs:EcoDispatchEnv" --store --steps 2000000 --test-ste 200 --test-inter 50000 --path "data/final_experiments/eco_dispatch/replace_min/" --num 10 --hyper "{'batch_size': 1024, 'memory_size': 1000000, 'actor_fc_dims': (256, 256, 256, 256, 256, 256), 'critic_fc_dims': (256, 256, 256, 256, 256, 256)}" --env-hyper "{'reward_function': 'replacement'}"
python src/main.py --agent "Ddpg1Step" --enviro "mlopf.envs.thesis_envs:EcoDispatchEnv" --store --steps 2000000 --test-ste 200 --test-inter 50000 --path "data/final_experiments/eco_dispatch/replace_mean/" --num 10 --hyper "{'batch_size': 1024, 'memory_size': 1000000, 'actor_fc_dims': (256, 256, 256, 256, 256, 256), 'critic_fc_dims': (256, 256, 256, 256, 256, 256)}" --env-hyper "{'reward_function': 'replacementA'}"
