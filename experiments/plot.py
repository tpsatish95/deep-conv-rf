import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import run

sns.set()
plt.style.use('seaborn')

###############################################################################
# Config
###############################################################################

plot_title = run.TITLE
base_path = run.RESULTS_PATH
results_path = run.RESULTS_PATH
x_lables = run.number_of_train_samples_space

###############################################################################
# Data
###############################################################################


def load_results(file_name):
    file_name = results_path + file_name + ".npy"
    if os.path.exists(file_name):
        return list(zip(*np.load(file_name)))
    else:
        return list([[np.nan]*len(x_lables), [np.nan]*len(x_lables)])


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


###############################################################################
# Plot Accuracies
###############################################################################

fig, ax = plt.subplots()
ax.plot(x_lables, naive_rf_acc_vs_n, marker="", color='green',
        linestyle=":", label="NaiveRF")

# ax.plot(x_lables, naive_rf_pyrerf_acc_vs_n, marker="",
#         color='black', linestyle=":", label="Naive RF (pyrerf)")

ax.plot(x_lables, deep_conv_rf_old_acc_vs_n, marker="", color='brown',
        linestyle="--", label="DeepConvRF (1-layer, unshared)")
ax.plot(x_lables, deep_conv_rf_old_two_layer_acc_vs_n, marker="",
        color='brown', label="DeepConvRF (2-layer, unshared)")

ax.plot(x_lables, deep_conv_rf_acc_vs_n, marker="", color='green',
        linestyle="--", label="DeepConvRF (1-layer, shared)")
ax.plot(x_lables, deep_conv_rf_two_layer_acc_vs_n, marker="",
        color='green', label="DeepConvRF (2-layer, shared)")

# ax.plot(x_lables, deep_conv_rf_pyrerf_acc_vs_n, marker="", linestyle="--",
#         color='black', label="DeepConvRF (1-layer, shared, pyrerf)")
# ax.plot(x_lables, deep_conv_rf_pyrerf_two_layer_acc_vs_n_times, marker="",
#         color='black', label="DeepConvRF (2-layer, shared, pyrerf)")

ax.plot(x_lables, np.array(cnn_acc_vs_n)/100.0, marker="", color='orange',
        linestyle=":", label="CNN (1-layer, 1-filter)")
ax.plot(x_lables, np.array(cnn32_acc_vs_n)/100.0, marker="", color='orange',
        linestyle="--", label="CNN (1-layer, 32-filters)")
ax.plot(x_lables, np.array(cnn32_two_layer_acc_vs_n)/100.0, marker="",
        color='orange', label="CNN (2-layer, 32-filters)")

ax.plot(x_lables, np.array(cnn_best_acc_vs_n)/100.0,
        marker="", color='blue', label="CNN (ResNet18)")


ax.set_xlabel('# of Train Samples', fontsize=18)
ax.set_xscale('log')
ax.set_xticks(x_lables)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax.set_ylabel('Accuracy', fontsize=18)

ax.set_title(plot_title + " Classification", fontsize=18)
plt.legend()
plt.savefig(base_path + "accuracy_comparisons.png")


###############################################################################
# Plot Execution Times
###############################################################################

fig, ax = plt.subplots()
ax.plot(x_lables, naive_rf_acc_vs_n_times, marker="", color='green',
        linestyle=":", label="NaiveRF")

# ax.plot(x_lables, naive_rf_pyrerf_acc_vs_n_times, marker="",
#         color='black', linestyle=":", label="Naive RF (pyrerf)")

ax.plot(x_lables, deep_conv_rf_old_acc_vs_n_times, marker="", color='brown',
        linestyle="--", label="DeepConvRF (1-layer, unshared)")
ax.plot(x_lables, deep_conv_rf_old_two_layer_acc_vs_n_times, marker="",
        color='brown', label="DeepConvRF (2-layer, unshared)")

ax.plot(x_lables, deep_conv_rf_acc_vs_n_times, marker="", color='green',
        linestyle="--", label="DeepConvRF (1-layer, shared)")
ax.plot(x_lables, deep_conv_rf_two_layer_acc_vs_n_times, marker="",
        color='green', label="DeepConvRF (2-layer, shared)")

# ax.plot(x_lables, deep_conv_rf_pyrerf_acc_vs_n_times, marker="", linestyle="--",
#         color='black', label="DeepConvRF (1-layer, shared, pyrerf)")
# ax.plot(x_lables, deep_conv_rf_pyrerf_acc_vs_n_times, marker="",
#         color='black', label="DeepConvRF (1-layer, shared, pyrerf)")

ax.plot(x_lables, cnn_acc_vs_n_times, marker="", color='orange',
        linestyle=":", label="CNN (1-layer, 1-filter)")
ax.plot(x_lables, cnn32_acc_vs_n_times, marker="", color='orange',
        linestyle="--", label="CNN (1-layer, 32-filters)")
ax.plot(x_lables, cnn32_two_layer_acc_vs_n_times, marker="",
        color='orange', label="CNN (2-layer, 32-filters)")

ax.plot(x_lables, cnn_best_acc_vs_n_times, marker="", color='blue', label="CNN (ResNet18)")


ax.set_xlabel('# of Train Samples', fontsize=18)
ax.set_xscale('log')
ax.set_xticks(x_lables)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax.set_ylabel('Execution Time (seconds)', fontsize=18)

ax.set_title(plot_title + " Classification Performance", fontsize=18)
plt.legend()
plt.savefig(base_path + "perf_comparisons.png")
