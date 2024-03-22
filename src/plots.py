from collections import defaultdict
import os

import matplotlib.pyplot as plt
import pandas as pd

from drl.util.plot_returns import get_algo_plus_hyperparams


WINDOW = 5


def main(path: str):
    base_results = get_results(path + '20230712_qmarket_baseline/')
    print(base_results)
    qmarket_obs(path, base_results)


def get_results(path: str):
    run_paths = os.listdir(path)

    results = defaultdict(list)
    for run_path in run_paths:
        res = pd.read_csv(path + run_path + '/test_returns.csv', index_col=0)
        hyperparams = get_hyperparams(path + run_path)

        print(hyperparams)

        results[hyperparams].append(res)

    results = {hp: sum(res) / len(res) for hp, res in results.items()}

    if len(results) == 1:
        return tuple(results.values())[0]

    return results


def qmarket_obs(path, base_results):

    obs_results = get_results(path + '20230712_qmarket_obs/')

    x = base_results.index

    y = base_results.valid_mape.rolling(window=WINDOW).mean()
    plt.plot(x, y, label='Baseline')

    for hp, data in obs_results.items():
        y = data.valid_mape.rolling(window=WINDOW).mean()
        plt.plot(x, y, label=hp)

    plt.legend()

    plt.show()


def get_hyperparams(path):
    with open(path + '/meta-data.txt') as f:
        lines = f.readlines()
    env_hyperparams = lines[7][25:-1]

    return env_hyperparams


if __name__ == '__main__':
    main('HPC/drlopf_experiments/data/final_experiments/')
