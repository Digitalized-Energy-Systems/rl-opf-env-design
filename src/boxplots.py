
import json

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


SHOWFLIERS = True  # False Required for reward boxplots 


def main():
    path = 'compact_results/'

    with open(path + 'mpe.json', 'r') as f:
        mpes = json.load(f)
    with open(path + 'invalid.json', 'r') as f:
        invalids = json.load(f)

    base_mpe_qmarket = mpes['qmarket_{}_{}']
    base_mpe_eco = mpes['eco_{}_{}']

    base_invalid_qmarket = invalids['qmarket_{}_{}']
    base_invalid_eco = invalids['eco_{}_{}']

    # Create data boxplots
    mpe_data = [
        [base_mpe_qmarket, mpes["qmarket_{'train_data': 'full_uniform'}_{}"], mpes["qmarket_{'train_data': 'normal_around_mean', 'sampling_kwargs': {'std': 0.3}}_{}"]], 
        # TODO: strange different std for eco?!
        [base_mpe_eco, mpes["eco_{'train_data': 'full_uniform'}_{}"], mpes["eco_{'train_data': 'normal_around_mean'}_{}"]]  
    ] 
    invalid_data = [
        [base_invalid_qmarket, invalids["qmarket_{'train_data': 'full_uniform'}_{}"], invalids["qmarket_{'train_data': 'normal_around_mean', 'sampling_kwargs': {'std': 0.3}}_{}"]], 
        # TODO: strange different std for eco?!
        [base_invalid_eco, invalids["eco_{'train_data': 'full_uniform'}_{}"], invalids["eco_{'train_data': 'normal_around_mean'}_{}"]]
    ] 
    create_boxplot_subfigs(mpe_data, invalid_data, ['Time-Series', 'Uniform', 'Normal'], path, 'data')

    # Create obs boxplots
    mpe_data = [
        [base_mpe_qmarket, mpes["qmarket_{'add_res_obs': True}_{}"], mpes["qmarket_{'add_res_obs': True, 'add_act_obs': True}_{}"]],
        [base_mpe_eco, mpes["eco_{'add_res_obs': True}_{}"], mpes["eco_{'add_res_obs': True, 'add_act_obs': True}_{}"]]
    ]
    invalid_data = [
        [base_invalid_qmarket, invalids["qmarket_{'add_res_obs': True}_{}"], invalids["qmarket_{'add_res_obs': True, 'add_act_obs': True}_{}"]],
        [base_invalid_eco, invalids["eco_{'add_res_obs': True}_{}"], invalids["eco_{'add_res_obs': True, 'add_act_obs': True}_{}"]]
    ]
    create_boxplot_subfigs(mpe_data, invalid_data, ['Markov', 'Red. (Fixed)', 'Red. (Random)'], path, 'obs')

    # Create episode boxplots
    mpe_data = [
        [mpes["qmarket_{'add_res_obs': True, 'add_act_obs': True}_{}"], mpes["qmarket_{'add_res_obs': True, 'add_act_obs': True, 'steps_per_episode': 5}_{'gamma': 0.5}"], mpes["qmarket_{'add_res_obs': True, 'add_act_obs': True, 'steps_per_episode': 5}_{'gamma': 0.9}"]],
        [mpes["eco_{'add_res_obs': True, 'add_act_obs': True}_{}"], mpes["eco_{'add_res_obs': True, 'add_act_obs': True, 'steps_per_episode': 5}_{'gamma': 0.5}"], mpes["eco_{'add_res_obs': True, 'add_act_obs': True, 'steps_per_episode': 5}_{'gamma': 0.9}"]]
    ]
    invalid_data = [
        [invalids["qmarket_{'add_res_obs': True, 'add_act_obs': True}_{}"], invalids["qmarket_{'add_res_obs': True, 'add_act_obs': True, 'steps_per_episode': 5}_{'gamma': 0.5}"], invalids["qmarket_{'add_res_obs': True, 'add_act_obs': True, 'steps_per_episode': 5}_{'gamma': 0.9}"]],
        [invalids["eco_{'add_res_obs': True, 'add_act_obs': True}_{}"], invalids["eco_{'add_res_obs': True, 'add_act_obs': True, 'steps_per_episode': 5}_{'gamma': 0.5}"], invalids["eco_{'add_res_obs': True, 'add_act_obs': True, 'steps_per_episode': 5}_{'gamma': 0.9}"]]
    ]
    create_boxplot_subfigs(mpe_data, invalid_data, ['1-step*', 'n-step (0.5)', 'n-step (0.9)'], path, 'episode')

    # Create reward fct boxplots
    mpe_data = [
        [base_mpe_qmarket, mpes["qmarket_{'reward_function': 'summation', 'ext_grid_pen_kwargs': {'linear_penalty': 5000}}_{}"], mpes["qmarket_{'reward_function': 'replacement'}_{}"], mpes["qmarket_{'reward_function': 'replacementA'}_{}"]],
        [base_mpe_eco, mpes["eco_{'reward_function': 'summation', 'ext_grid_pen_kwargs': {'linear_penalty': 100000}}_{}"], mpes["eco_{'reward_function': 'replacement'}_{}"], mpes["eco_{'reward_function': 'replacementA'}_{}"]]
    ]
    invalid_data = [
        [base_invalid_qmarket, invalids["qmarket_{'reward_function': 'summation', 'ext_grid_pen_kwargs': {'linear_penalty': 5000}}_{}"], invalids["qmarket_{'reward_function': 'replacement'}_{}"], invalids["qmarket_{'reward_function': 'replacementA'}_{}"]],
        [base_invalid_eco, invalids["eco_{'reward_function': 'summation', 'ext_grid_pen_kwargs': {'linear_penalty': 100000}}_{}"], invalids["eco_{'reward_function': 'replacement'}_{}"], invalids["eco_{'reward_function': 'replacementA'}_{}"]]
    ]
    create_boxplot_subfigs(mpe_data, invalid_data, ['Sum', 'Sum (10x)', 'Repl. (Mean)', 'Repl. (Min)'], path, 'reward')


def create_boxplot_subfigs(mpe_data, invalid_data, labels, path, name):
    """ Create 4 subplots with """
    fig, axs = plt.subplots(2, 2, figsize=(12, 5))
    # Reduce horizontal space between subplots
    plt.subplots_adjust(wspace=0.15)
    axs[0, 0].boxplot(mpe_data[0], labels=labels)
    axs[0, 1].boxplot(mpe_data[1], labels=labels)
    axs[1, 0].boxplot(invalid_data[0], labels=labels)
    if len(labels) == 4:
        labels[3] = labels[3] + '*'
    axs[1, 1].boxplot(invalid_data[1], labels=labels, showfliers=SHOWFLIERS)

    axs[0, 0].set_ylabel('MAPE in %')
    axs[1, 0].set_ylabel('Invalid Share in %')

    axs[0, 0].set_title('VoltageControl')
    axs[0, 1].set_title('EcoDispatch')

    print(name)
    print(f'MPE median qmarket: {[round(np.median(data), 3) for data in mpe_data[0]]}')
    print(f'MPE median eco: {[round(np.median(data), 3) for data in mpe_data[1]]}')
    print(f'Invalid median qmarket: {[round(np.median(data), 3) for data in invalid_data[0]]}')
    print(f'Invalid median eco: {[round(np.median(data), 3) for data in invalid_data[1]]}')
    print('')

    plt.savefig(path + f'{name}.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close()


def create_boxplot(data, labels, ylabel, path):
    plt.figure(figsize=(2*3.15, 2*1.97))
    plt.boxplot(data, labels=labels)
    plt.ylabel(ylabel)
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
