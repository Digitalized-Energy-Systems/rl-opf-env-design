import matplotlib.pyplot as plt
import numpy as np

from mlopf.envs.thesis_envs import QMarketEnv, EcoDispatchEnv


def scatter_plot(env, n_samples=1000, normalize=True):
    """ Plot the objective and penalty values for a given number of samples. """
    objs = []
    viol = []
    for _ in range(n_samples):
        env._sampling(step=None, test=False, sample_new=True)
        env._apply_actions(env.action_space.sample())
        env._run_pf()
        objs.append(np.sum(env.calc_objective(env.net)))
        viol.append(np.sum(env.calc_violations()[1]))

    objs = -np.array(objs)
    viol = np.array(viol)

    print(f'Min objective: {objs.min()}')
    print(f'Max objective: {objs.max()}')
    print(f'Min violation: {viol.min()}')
    print(f'Max violation: {viol.max()}')
    print(f'Mean objective: {objs.mean()}')
    print(f'Mean violation: {viol.mean()}')
    print(f'Std objective: {np.std(objs)}')
    print(f'Std violation: {np.std(viol)}')
    print(f'Correlation coefficient: {np.corrcoef(objs, viol)[0][1]}')

    if normalize:
        objs = (objs - objs.min()) / (objs.max() - objs.min())
        viol = (viol - viol.min()) / (viol.max() - viol.min())

    plt.figure(figsize=(6.3, 3.5))
    plt.scatter(objs, viol)
    plt.xlabel('Objective J(s)')
    plt.ylabel('Sum of Violations')
    plt.savefig(f'{env.__class__.__name__}_scatter.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()


if __name__ == '__main__':
    print('Eco Dispatch Environment:')
    env = EcoDispatchEnv()
    scatter_plot(env)
    print('--------------------------')
    print('Voltage Control Environment:')
    env = QMarketEnv()
    scatter_plot(env)
