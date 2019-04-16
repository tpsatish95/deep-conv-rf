# general imports
import copy
import os.path
import time
import warnings

import numpy as np
import torch

from cnn.models.resnet import ResNet18
from cnn.models.simple import SimpleCNN1layer, SimpleCNN2Layers
from cnn.trainer import run_cnn
from dataset import get_dataset
from random_forest.deep_conv_rf_shared import (
    run_one_layer_deep_conv_rf_shared, run_two_layer_deep_conv_rf_shared)
from random_forest.deep_conv_rf_unshared import (
    run_one_layer_deep_conv_rf_unshared, run_two_layer_deep_conv_rf_unshared)
from random_forest.naive_rf import run_naive_rf

warnings.filterwarnings("ignore")

##############################################################################################################
# Settings
##############################################################################################################

DATASET_NAME = "CIFAR10"
DATA_PATH = "./data"

RESULTS_PATH = "results/cifar10/100_percent_data/2vs4/"

CHOOSEN_CLASSES = [2, 4]
NUM_CLASSES = 2
MAX_TRAIN_FRACTION = 1.0

BATCH_SIZE = 128
EPOCH = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CNN_CONFIG = {"batch_size": BATCH_SIZE, "epoch": EPOCH, "device": DEVICE}

print("Are GPUs available? " + str(torch.cuda.is_available()) + "\n")

##############################################################################################################
# Data
##############################################################################################################

fraction_of_train_samples_space = np.geomspace(0.01, MAX_TRAIN_FRACTION, num=10)

numpy_data = dict()
(numpy_data["train_images"], numpy_data["train_labels"]), (numpy_data["test_images"], numpy_data["test_labels"]) = \
    get_dataset(DATA_PATH, DATASET_NAME, is_numpy=True)

pytorch_data = dict()
pytorch_data["trainset"], pytorch_data["testset"] = get_dataset(DATA_PATH, DATASET_NAME, is_numpy=False)


##############################################################################################################
# Helpers
##############################################################################################################


def print_old_results(file_name):
    global fraction_of_train_samples_space

    accuracy_scores = np.load(file_name)

    for fraction_of_train_samples, (best_accuracy, time_taken) in zip(fraction_of_train_samples_space, accuracy_scores):
        print("Train Fraction:", str(fraction_of_train_samples))
        print("Accuracy:", str(best_accuracy))
        print("Experiment Runtime: " + str(time_taken), "\n")
        print()
    return accuracy_scores


def run_experiment(experiment, results_file_name, experiment_name, repeats=2, cnn_model=None, cnn_config={}):
    global fraction_of_train_samples_space, numpy_data, pytorch_data

    print("##################################################################")
    print("acc vs n_samples: " + experiment_name + "\n")

    acc_vs_n = list()
    file_name = RESULTS_PATH + results_file_name + ".npy"

    if not os.path.exists(file_name):
        for fraction_of_train_samples in fraction_of_train_samples_space:

            if not cnn_model:
                start = time.time()
                best_accuracy = np.mean(
                    [experiment(numpy_data, CHOOSEN_CLASSES, fraction_of_train_samples) for _ in range(repeats)])
                end = time.time()

            else:
                start = time.time()
                best_accuracy = np.mean(
                    [experiment(cnn_model, pytorch_data, CHOOSEN_CLASSES, fraction_of_train_samples, cnn_config) for _ in range(repeats)])
                end = time.time()

            time_taken = (end - start)/float(repeats)
            acc_vs_n.append((best_accuracy, time_taken))

            print("Train Fraction:", str(fraction_of_train_samples))
            print("Accuracy:", str(best_accuracy))
            print("Experiment Runtime: " + str(time_taken), "\n")

        np.save(file_name, acc_vs_n)

    else:
        acc_vs_n = print_old_results(file_name)

    print("##################################################################")

    return acc_vs_n


##############################################################################################################
# Runners
##############################################################################################################

if __name__ == '__main__':

    script_start = time.time()

    run_experiment((run_naive_rf, "naive_rf_acc_vs_n", "Naive RF"))

    # DeepConvRF Unshared
    run_experiment((run_one_layer_deep_conv_rf_unshared, "deep_conv_rf_old_acc_vs_n", "DeepConvRF (1-layer, unshared)"))
    run_experiment(run_two_layer_deep_conv_rf_unshared, "deep_conv_rf_old_two_layer_acc_vs_n", "DeepConvRF (2-layer, unshared)")

    # DeepConvRF Shared
    run_experiment(run_one_layer_deep_conv_rf_shared, "deep_conv_rf_acc_vs_n", "DeepConvRF (1-layer, shared)")
    run_experiment(run_two_layer_deep_conv_rf_shared, "deep_conv_rf_two_layer_acc_vs_n", "DeepConvRF (2-layer, shared)")

    # # DeepConvRerF Shared
    # naive_rf_pyrerf_acc_vs_n, naive_rf_pyrerf_acc_vs_n_times = list(zip(*np.load(base_path+"naive_rf_pyrerf_acc_vs_n.npy")))
    # deep_conv_rf_pyrerf_acc_vs_n, deep_conv_rf_pyrerf_acc_vs_n_times = list(zip(*np.load(base_path+"deep_conv_rf_pyrerf_acc_vs_n.npy")))

    # CNN
    cnn_acc_vs_n_config = copy.deepcopy(CNN_CONFIG)
    cnn_acc_vs_n_config.update({'model': 0, 'lr': 0.001, 'weight_decay': 1e-05})
    run_experiment(run_cnn, "cnn_acc_vs_n", "CNN (1-layer, 1-filter)", cnn_model=SimpleCNN1layer(1, NUM_CLASSES), cnn_config=cnn_acc_vs_n_config)

    cnn32_acc_vs_n_config = copy.deepcopy(CNN_CONFIG)
    cnn32_acc_vs_n_config.update({'model': 1, 'lr': 0.001, 'weight_decay': 0.1})
    run_experiment(run_cnn, "cnn32_acc_vs_n", "CNN (1-layer, 32-filter)", cnn_model=SimpleCNN1layer(32, NUM_CLASSES), cnn_config=cnn32_acc_vs_n_config)

    cnn32_two_layer_acc_vs_n_config = copy.deepcopy(CNN_CONFIG)
    cnn32_two_layer_acc_vs_n_config.update({'model': 2, 'lr': 0.001, 'weight_decay': 0.1})
    run_experiment(run_cnn, "cnn32_two_layer_acc_vs_n", "CNN (2-layer, 32-filter)", cnn_model=SimpleCNN2Layers(32, NUM_CLASSES), cnn_config=cnn32_two_layer_acc_vs_n_config)

    # Best CNN
    run_experiment(run_cnn, "cnn_best_acc_vs_n", "CNN (ResNet18)", cnn_model=ResNet18(NUM_CLASSES), cnn_config=CNN_CONFIG)

    script_end = time.time()
    print("Total Runtime: " + str(script_end - script_start), "\n")
