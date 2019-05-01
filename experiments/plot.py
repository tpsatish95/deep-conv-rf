import copy
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import sem

import run

sns.set()
plt.style.use('seaborn')

###############################################################################
# Config
###############################################################################

plot_title = run.TITLE
results_path = run.RESULTS_PATH
x_lables = run.number_of_train_samples_space
n_trials = run.N_TRIALS

###############################################################################
# Data
###############################################################################


def load_results(file_name):
    file_name = results_path + file_name + ".npy"
    if os.path.exists(file_name):
        return list(zip(*[list(zip(*i)) for i in list(zip(*np.load(file_name)))]))
    else:
        return list([[[np.nan]*n_trials]*len(x_lables), [[np.nan]*n_trials]*len(x_lables)])


# Naive RF
naive_rf_acc_vs_n, naive_rf_acc_vs_n_times = load_results("naive_rf_acc_vs_n")

# # Naive RerF
# naive_rf_pyrerf_acc_vs_n, naive_rf_pyrerf_acc_vs_n_times = load_results("naive_rf_pyrerf_acc_vs_n")

# DeepConvRF Unshared
deep_conv_rf_old_acc_vs_n, deep_conv_rf_old_acc_vs_n_times = load_results(
    "deep_conv_rf_old_acc_vs_n")
deep_conv_rf_old_two_layer_acc_vs_n, deep_conv_rf_old_two_layer_acc_vs_n_times = load_results(
    "deep_conv_rf_old_two_layer_acc_vs_n")

# DeepConvRF Shared
deep_conv_rf_acc_vs_n, deep_conv_rf_acc_vs_n_times = load_results("deep_conv_rf_acc_vs_n")
deep_conv_rf_two_layer_acc_vs_n, deep_conv_rf_two_layer_acc_vs_n_times = load_results(
    "deep_conv_rf_two_layer_acc_vs_n")

# # DeepConvRF Shared (pyrerf)
# deep_conv_rf_pyrerf_acc_vs_n, deep_conv_rf_pyrerf_acc_vs_n_times = load_results(
#     "deep_conv_rf_pyrerf_acc_vs_n")
# deep_conv_rf_pyrerf_two_layer_acc_vs_n, deep_conv_rf_pyrerf_two_layer_acc_vs_n_times = load_results(
#     "deep_conv_rf_pyrerf_two_layer_acc_vs_n")

# CNN
cnn_acc_vs_n, cnn_acc_vs_n_times = load_results("cnn_acc_vs_n")
cnn32_acc_vs_n, cnn32_acc_vs_n_times = load_results("cnn32_acc_vs_n")
cnn32_two_layer_acc_vs_n, cnn32_two_layer_acc_vs_n_times = load_results("cnn32_two_layer_acc_vs_n")

# Best CNN
cnn_best_acc_vs_n, cnn_best_acc_vs_n_times = load_results("cnn_best_acc_vs_n")


###############################################################################
# Plot Settings
###############################################################################

plt.rcParams['figure.figsize'] = 15, 12
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['font.size'] = 25
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.handlelength'] = 3
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['lines.linewidth'] = 3

experiment_plot_styles = {
    "naive_rf_acc_vs_n": {"marker": "", "color": "green", "linestyle": ":", "label": "NaiveRF"},
    # "naive_rf_pyrerf_acc_vs_n": {"marker": "", "color": "black", "linestyle": ":", "label": "Naive RF (pyrerf)"},

    "deep_conv_rf_old_acc_vs_n": {"marker": "", "color": "brown", "linestyle": "--", "label": "DeepConvRF (1-layer, unshared)"},
    "deep_conv_rf_old_two_layer_acc_vs_n": {"marker": "", "color": "brown", "label": "DeepConvRF (2-layer, unshared)"},

    "deep_conv_rf_acc_vs_n": {"marker": "", "color": "green", "linestyle": "--", "label": "DeepConvRF (1-layer, shared)"},
    "deep_conv_rf_two_layer_acc_vs_n": {"marker": "", "color": "green", "label": "DeepConvRF (2-layer, shared)"},

    # "deep_conv_rf_pyrerf_acc_vs_n": {"marker": "", "color": "black", "linestyle": "--", "label": "DeepConvRF (1-layer, shared, pyrerf)"},
    # "deep_conv_rf_pyrerf_two_layer_acc_vs_n": {"marker": "", "color": "black", "label": "DeepConvRF (2-layer, shared, pyrerf)"},

    "cnn_acc_vs_n": {"marker": "", "color": "orange", "linestyle": ":", "label": "CNN (1-layer, 1-filter)"},
    "cnn32_acc_vs_n": {"marker": "", "color": "orange", "linestyle": "--", "label": "CNN (1-layer, 32-filter)"},

    "cnn32_two_layer_acc_vs_n": {"marker": "", "color": "orange", "label": "CNN (2-layer, 32-filter)"},

    "cnn_best_acc_vs_n": {"marker": "", "color": "blue", "label": "CNN (ResNet18)"}
}

###############################################################################
# Plot Helpers
###############################################################################


def plot_experiment(plot_ax, x, experiment_name, plot_params, is_performance=False, plot_all_trials=True, plot_error_bars=True):
    plot_params = copy.deepcopy(plot_params)
    if not is_performance:
        trials = np.array(eval(experiment_name))
    else:
        trials = np.array(eval(experiment_name + "_times"))

    if plot_error_bars:
        plot_ax.errorbar(x, np.mean(trials, axis=1), yerr=sem(trials, axis=1), elinewidth=2, **plot_params)
    else:
        plot_ax.plot(x, np.mean(trials, axis=1), **plot_params)

    if plot_all_trials:
        del plot_params["label"]
        for trial_number in range(trials.shape[1]):
            plot_ax.plot(x, trials[:, trial_number], alpha=0.4, **plot_params)


def plot_experiments(title, experiments, save_to, is_performance=False, plot_all_trials=True, plot_error_bars=True):
    global experiment_plot_styles, x_lables

    fig, ax = plt.subplots()

    for experiment_name in experiments:
        plot_experiment(ax, x_lables, experiment_name,
                        experiment_plot_styles[experiment_name], is_performance=is_performance, plot_all_trials=plot_all_trials)

    ax.set_xlabel('# of Train Samples', fontsize=18)
    ax.set_xscale('log')
    ax.set_xticks(x_lables)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    if not is_performance:
        ax.set_ylabel('Accuracy', fontsize=18)
    else:
        ax.set_ylabel('Execution Time (seconds)', fontsize=18)

    ax.set_title(plot_title + " " + title, fontsize=18)
    plt.legend()

    plt.savefig(results_path + save_to + ".png")


###############################################################################
# Plot Groups
###############################################################################

all_experiments = [
    "naive_rf_acc_vs_n",
    "deep_conv_rf_old_acc_vs_n",
    "deep_conv_rf_old_two_layer_acc_vs_n",
    "deep_conv_rf_acc_vs_n",
    "deep_conv_rf_two_layer_acc_vs_n",
    "cnn_acc_vs_n",
    "cnn32_acc_vs_n",
    "cnn32_two_layer_acc_vs_n",
    "cnn_best_acc_vs_n"
]

one_layer_experiments = [
    "naive_rf_acc_vs_n",
    "deep_conv_rf_old_acc_vs_n",
    "deep_conv_rf_acc_vs_n",
    "cnn_acc_vs_n",
    "cnn32_acc_vs_n",
    "cnn_best_acc_vs_n"
]

n_layer_experiments = [
    "naive_rf_acc_vs_n",
    "deep_conv_rf_old_two_layer_acc_vs_n",
    "deep_conv_rf_two_layer_acc_vs_n",
    "cnn32_two_layer_acc_vs_n",
    "cnn_best_acc_vs_n"
]


###############################################################################
# Plot
###############################################################################
plot_experiments("Classification (1-layer)", one_layer_experiments,
                 plot_all_trials=False, plot_error_bars=True, save_to="accuracy_comparisons_1_layer")
plot_experiments("Classification (n-layers)", n_layer_experiments,
                 plot_all_trials=False, plot_error_bars=True, save_to="accuracy_comparisons_n_layer")
plot_experiments("Classification", all_experiments,
                 plot_all_trials=False, plot_error_bars=True, save_to="accuracy_comparisons")

plot_experiments("Classification Performance", all_experiments, is_performance=True,
                 plot_all_trials=False, plot_error_bars=False, save_to="perf_comparisons")
