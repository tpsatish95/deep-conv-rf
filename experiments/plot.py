import copy
import os
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import sem

from dataset import get_dataset
from utils import get_number_of_train_samples_space, get_title_and_results_path

sns.set()
plt.style.use('seaborn')

DATASETS = dict()
DATA_PATH = "./data"

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


def load_results(file_name, results_path, x_lables, n_trials, with_time_tracker=False):
    '''
    time_tracker dict's items:
        load
          - train_chop
          - test_chop
        train
          - fit
          - train_post
          - final_fit
        test
          - test_post
          - final_predict
    '''
    file_name = results_path + file_name + ".npy"
    if os.path.exists(file_name):
        file_obj = np.load(file_name)
        if not with_time_tracker:
            return list(zip(*[list(zip(*i)) for i in list(zip(*file_obj[:2]))]))
        else:
            return list(zip(*[list(zip(*i)) for i in list(zip(*file_obj[:2]))])) + [file_obj[2]]

    else:
        if not with_time_tracker:
            return list([[[np.nan]*n_trials]*len(x_lables), [[np.nan]*n_trials]*len(x_lables)])
        else:
            return list([[[np.nan]*n_trials]*len(x_lables), [[np.nan]*n_trials]*len(x_lables), [{"train": [np.nan]*n_trials, "test": [np.nan]*n_trials}]*len(x_lables)])


def plot_experiment(plot_ax, x, n_trials, experiment_name, plot_params, results_path, is_performance=False, plot_all_trials=True, plot_error_bars=True):
    plot_params = copy.deepcopy(plot_params)
    if not is_performance:
        trials = np.array(load_results(experiment_name, results_path, x, n_trials)[0])
    else:
        trials = np.array(load_results(experiment_name, results_path, x, n_trials)[1])

    if plot_error_bars and not is_performance:
        plot_ax.errorbar(x, np.mean(trials, axis=1), yerr=sem(
            trials, axis=1), elinewidth=2, **plot_params)
    else:
        plot_ax.plot(x, np.mean(trials, axis=1), **plot_params)

    if plot_all_trials:
        del plot_params["label"]
        for trial_number in range(trials.shape[1]):
            plot_ax.plot(x, trials[:, trial_number], alpha=0.4, **plot_params)


def plot_experiments(title, experiments, config, save_to, is_performance=False, plot_all_trials=True, plot_error_bars=True):
    global experiment_plot_styles, DATASETS

    plot_title, results_path = get_title_and_results_path(
        config["dataset_name"], config["choosen_classes"], config["min_samples"], config["max_samples"])

    if config["dataset_name"] not in DATASETS:
        DATASETS[config["dataset_name"]] = get_dataset(
            DATA_PATH, config["dataset_name"], is_numpy=True)
    x_lables = get_number_of_train_samples_space(
        DATASETS[config["dataset_name"]], config["choosen_classes"], config["min_samples"], config["max_samples"])
    n_trials = config["n_trials"]

    fig, ax = plt.subplots()

    for experiment_name in experiments:
        plot_experiment(ax, x_lables, n_trials, experiment_name,
                        experiment_plot_styles[experiment_name], results_path, is_performance=is_performance, plot_all_trials=plot_all_trials)

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

    plt.clf()
    plt.close(fig)


def plot_all_figures(config):
    plot_experiments("Classification (1-layer)", one_layer_experiments, config,
                     plot_all_trials=False, plot_error_bars=True, save_to="accuracy_comparisons_1_layer")
    plot_experiments("Classification (n-layers)", n_layer_experiments, config,
                     plot_all_trials=False, plot_error_bars=True, save_to="accuracy_comparisons_n_layer")
    plot_experiments("Classification", all_experiments, config,
                     plot_all_trials=False, plot_error_bars=True, save_to="accuracy_comparisons")

    plot_experiments("Classification Performance", all_experiments, config, is_performance=True,
                     plot_all_trials=False, plot_error_bars=False, save_to="perf_comparisons")


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

if __name__ == '__main__':
    config = json.load(open("config.json", "r"))
    # plot_all_figures(config["CIFAR10"]["1vs9"]["10_to_100"])

    # plot all:
    for dataset in config:
        for pairs in config[dataset]:
            for sample_range in config[dataset][pairs]:
                print(config[dataset][pairs][sample_range])
                plot_all_figures(config[dataset][pairs][sample_range])
