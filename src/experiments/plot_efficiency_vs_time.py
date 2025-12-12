import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_efficiency_tensor(metrics_path):
    with open(metrics_path, 'rb') as handle:
        metrics = pickle.load(handle)
    # Expect key 'nudge_efficiency' -> shape [seed, user, time]
    eff = metrics.get('nudge_efficiency', None)
    if eff is None:
        raise ValueError(f"Missing 'nudge_efficiency' in {metrics_path}")
    return np.asarray(eff)


def aggregate_efficiency(eff_tensor):
    # eff_tensor: [num_seeds, num_users, num_times]
    # mean over seeds and users; also compute SEM across seeds (after averaging users per seed)
    if eff_tensor.ndim != 3:
        raise ValueError("Efficiency tensor must be 3D: [seed, user, time]")
    # First average across users within each seed to get [seed, time]
    eff_seed_mean = np.mean(eff_tensor, axis=1)
    mean_over_seeds = np.mean(eff_seed_mean, axis=0)
    # Standard error over seeds
    num_seeds = eff_seed_mean.shape[0]
    sem_over_seeds = np.std(eff_seed_mean, axis=0, ddof=1) / max(1, np.sqrt(num_seeds))
    return mean_over_seeds, sem_over_seeds


def plot_efficiency(alg_to_series, out_path):
    plt.figure(figsize=(8, 4.5))
    for alg_name, (mean_series, sem_series) in alg_to_series.items():
        t = np.arange(len(mean_series))
        plt.plot(t, mean_series, label=alg_name)
        # Optional shaded SEM
        plt.fill_between(t, mean_series - sem_series, mean_series + sem_series, alpha=0.2)

    plt.xlabel('user_time_t')
    plt.ylabel('Efficiency (cum DQ / cum nudges)')
    plt.title('Nudge Efficiency vs Time')
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main():
    # Base results directory path pattern (update if needed)
    base_dir = os.path.join('/tmp/budgeted_oral_results', 'eval_pooling', 'no_pooling_online_None')
    target_algs = ['DABBI_Nudging_I', 'DABBI_Nudging_II', 'UCB_UniformNudge']

    alg_to_series = {}
    for alg in target_algs:
        metrics_path = os.path.join(base_dir, f'{alg}_comprehensive_metrics.p')
        if not os.path.exists(metrics_path):
            print(f"Skipping {alg}: metrics file not found at {metrics_path}")
            continue
        eff_tensor = load_efficiency_tensor(metrics_path)
        mean_series, sem_series = aggregate_efficiency(eff_tensor)
        alg_to_series[alg] = (mean_series, sem_series)

    if len(alg_to_series) == 0:
        print('No metrics found to plot.')
        return

    out_fig = os.path.join(os.path.dirname(os.path.dirname(base_dir)), '..', 'figs', 'nudge_efficiency_vs_time.pdf')
    out_fig = os.path.abspath(out_fig)
    plot_efficiency(alg_to_series, out_fig)
    print(f'Saved figure to: {out_fig}')


if __name__ == '__main__':
    main()


