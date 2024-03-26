""" Generate data for metrics calculation """

from collections import defaultdict
from datetime import datetime
import json
import os
import random
import warnings

import pandas as pd
import numpy as np
import torch

from drl.util.load_agent import load_agent


DEVICE = ('cuda:0' if torch.cuda.is_available() else 'cpu')
REMOVE_HP = ('batch_size', 'critic_fc_dims', 'actor_fc_dims', 'memory_size')
N_TEST = 999999


warnings.filterwarnings("ignore")


def main():
    base_path = 'results/'

    qmarket_paths = ('base', 'data', 'res_obs', 'act_obs', 'nstep_05', 
                     'nstep_09', 'sum_10x', 'replace_min', 'replace_mean')
    eco_paths = ('base', 'data', 'res_obs', 'act_obs', 'nstep_05', 
                 'nstep_09', 'sum_10x', 'replace_min', 'replace_mean')

    regret_results, mpe_results, invalid_results, viol_results, rel_viol_results = \
        defaultdict(list),defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    
    for exp_paths, env_name in zip(
            (qmarket_paths, eco_paths), ('voltage_control', 'eco_dispatch')):
        opt_results = None
        for exp_path in exp_paths:
            path = os.path.join(base_path, env_name, exp_path)
            print('Path: ', path)

            run_paths = os.listdir(path)
            for run_path in run_paths:
                full_path = os.path.join(path, run_path)
                print('Load new agent from ', run_path)
                try:
                    agent, env, _ = get_agent_data(full_path)
                except FileNotFoundError:
                    # Training probably not finished or interrupted
                    print(f'{full_path}: File not found')
                    continue
                _, hps, env_design = get_algo_plus_hps(full_path, REMOVE_HP)
                description = f'{env_name}_{env_design}_{hps}'

                if not opt_results:
                    opt_results = get_opt_performance(env)
                    opt_objs, possible = opt_results
                    print('Possible share: ', np.mean(possible))

                objs, viol, rel_viol, valids = get_performance(agent, env)

                filter = np.logical_and(valids, possible)
                valid_regret = calculate_regret(objs[filter], opt_objs[filter])
                valid_mpe = calculate_mpe(valid_regret, opt_objs[filter])
                valid_share = calculate_valid_share(valids, possible)

                # print(f'Regret: {np.mean(valid_regret)}')
                # print(f'MPE: {round(valid_mpe, 4)}')
                # print(f'Valid share: {valid_share}')

                regret_results[description].append(np.mean(valid_regret))
                mpe_results[description].append(valid_mpe)
                invalid_results[description].append(100 - valid_share)
                viol_results[description].append(np.mean(viol))
                rel_viol_results[description].append(np.mean(rel_viol))
                

            description = f'{env_name}_random'
            if description not in regret_results.keys():
                print('Random agent')
                objs, viol, rel_viol, valids = get_random_performance(env)
                filter = np.logical_and(valids, possible)
                valid_regret = calculate_regret(objs[filter], opt_objs[filter])
                valid_mpe = calculate_mpe(valid_regret, opt_objs[filter])
                valid_share = calculate_valid_share(valids, possible)
                print(f'Regret: {np.mean(valid_regret)}')
                print(f'MPE: {round(valid_mpe, 4)}')
                print(f'Invalid share: {100 - valid_share}')  
                print(f'Violations: {np.mean(viol)}')
                print(f'Percentage violations: {np.mean(rel_viol)}')
                
                regret_results[description].append(np.mean(valid_regret))
                mpe_results[description].append(valid_mpe)
                invalid_results[description].append(100 - valid_share)
                viol_results[description].append(np.mean(viol))
                rel_viol_results[description].append(np.mean(rel_viol))

            for description in regret_results.keys():
                print(description)
                print('N samples: ', len(regret_results[description]))
                print('Regret: ', round(np.mean(regret_results[description]), 5), ' +/- ', round(np.std(regret_results[description]), 5), f'(Median: {np.median(regret_results[description])})')
                print('MPE: ', round(np.mean(mpe_results[description]), 3), ' +/- ', round(np.std(mpe_results[description]), 3), f'(Median: {np.median(mpe_results[description])})')
                print('Invalid share: ', round(np.mean(invalid_results[description]), 5), ' +/- ', round(np.std(invalid_results[description]), 5), f'(Median: {np.median(np.array(invalid_results[description]))})')
                print('Violations: ', round(np.mean(viol_results[description]), 8), ' +/- ', round(np.std(viol_results[description]), 8), f'(Median: {np.median(viol_results[description])})')
                print('Percentage violations: ', round(np.mean(rel_viol_results[description]), 3), ' +/- ', round(np.std(rel_viol_results[description]), 3), f'(Median: {np.median(rel_viol_results[description])})')
                print('')

            results_path = os.path.join(base_path, 'compact_results')
            os.makedirs(results_path, exist_ok=True)
            time_stamp = datetime.now().strftime("%Y%m%d")
            results_subpath = os.path.join(results_path, time_stamp)
            os.makedirs(results_subpath, exist_ok=True)
            # Create JSON file with results
            with open(f'{results_subpath}/regret.json', 'w') as f:
                json.dump(dict(regret_results), f)
            with open(f'{results_subpath}/mpe.json', 'w') as f:
                json.dump(dict(mpe_results), f)
            with open(f'{results_subpath}/invalid.json', 'w') as f:
                json.dump(dict(invalid_results), f)
            with open(f'{results_subpath}/viol.json', 'w') as f:
                json.dump(dict(viol_results), f)
            with open(f'{results_subpath}/rel_viol.json', 'w') as f:
                json.dump(dict(rel_viol_results), f)   
   

def get_algo_plus_hps(path, remove_hyperparams: tuple=()):
    with open(path + '/meta-data.txt') as f:
        lines = f.readlines()
    algo = lines[2][15:]
    hyperparams = lines[6][23:]
    env_hyperparams = lines[7][25:]

    hp_dict = eval(hyperparams)
    ehp_dict = eval(env_hyperparams)
    for rhp in remove_hyperparams:
        hp_dict.pop(rhp, None)
        ehp_dict.pop(rhp, None)
    hyperparams = str(hp_dict)
    env_hyperparams = str(ehp_dict)

    return algo.replace('\n', ''), hyperparams.replace('\n', ''), env_hyperparams.replace('\n', '')


def get_agent_data(path):
    agent = load_agent(path)
    return agent, agent.env, agent.name


def get_opt_performance(env):
    opt_objs = []
    possible = []
    np.random.seed(42)
    random.seed(42)
    for step in env.unwrapped.test_steps:
        env.reset(options={'step': step, 'test': True})
        pos = env.unwrapped.baseline_reward()
        opt_obj = env.unwrapped.calc_objective(env.unwrapped.net).sum()
        opt_objs.append(opt_obj)
        possible.append(~np.isnan(pos))

        if len(possible) >= N_TEST:
            break

    return np.array(opt_objs), np.array(possible)


def get_performance(agent, env):
    objs = []
    violations = []
    rel_violations = []
    valids = []
    np.random.seed(42)
    random.seed(42)
    for step in env.unwrapped.test_steps:
        done = False
        obs, info = env.reset(options={'step': step, 'test': True})
        while not done:
            act = agent.test_act(agent.scale_obs(obs))
            obs, _, terminated, truncated, info = env.step(act)
            done = terminated or truncated

        objs.append(env.unwrapped.calc_objective(env.unwrapped.net).sum())
        violations.append(info['violations'].sum())
        rel_violations.append(info['percentage_violations'].sum())
        valids.append(info['valids'].all())

        if len(valids) >= N_TEST:
            break

    return np.array(objs), np.array(violations), np.array(rel_violations), np.array(valids)


def get_random_performance(env):
    """ Compute the performance of a random agent as a baseline. """
    objs = []
    violations = []
    rel_violations = []
    valids = []
    np.random.seed(42)
    random.seed(42)
    for step in env.unwrapped.test_steps:
        done = False
        obs, info = env.reset(options={'step': step, 'test': True})
        while not done:
            act = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(act)
            done = terminated or truncated

        objs.append(env.unwrapped.calc_objective(env.unwrapped.net).sum())
        violations.append(info['violations'].sum())
        rel_violations.append(info['percentage_violations'].sum())
        valids.append(info['valids'].all())

        if len(valids) >= N_TEST:
            break

    return np.array(objs), np.array(violations), np.array(rel_violations), np.array(valids)


def calculate_valid_share(valids, possible):
    return valids[possible].mean() * 100


def calculate_regret(objs, opt_objs):
    return opt_objs - objs


def calculate_mpe(valid_regret, valid_opt_objs):
    return (valid_regret / abs(valid_opt_objs)).mean() * 100


if __name__ == '__main__':
    main()
