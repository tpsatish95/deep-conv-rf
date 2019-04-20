import copy
import logging
import os.path
import time
import warnings

import numpy as np
import torch

from cnn.models.resnet import ResNet18
from cnn.models.simple import SimpleCNN1layer, SimpleCNN2Layers
from cnn.trainer import run_cnn
from dataset import get_dataset
# from random_forest.deep_conv_rerf_shared import (
#     run_one_layer_deep_conv_rerf_shared, run_two_layer_deep_conv_rerf_shared)
from random_forest.deep_conv_rf_shared import (
    run_one_layer_deep_conv_rf_shared, run_two_layer_deep_conv_rf_shared)
from random_forest.deep_conv_rf_unshared import (
    run_one_layer_deep_conv_rf_unshared, run_two_layer_deep_conv_rf_unshared)
# from random_forest.naive_rerf import run_naive_rerf
from random_forest.naive_rf import run_naive_rf

warnings.filterwarnings("ignore")

##############################################################################################################
# Settings
##############################################################################################################

####### CIFAR10 ########
DATASET_NAME = "CIFAR10"
TITLE = "Automobile (1) vs Truck(9)"

DATA_PATH = "./data"
RESULTS_PATH = "results/cifar10/100_percent_data/1vs9/"

CHOOSEN_CLASSES = [1, 9]
MAX_TRAIN_FRACTION = 1.0

# ####### SVHN ########
# DATASET_NAME = "SVHN"
# TITLE = "1 vs 7"
#
# DATA_PATH = "./data"
# RESULTS_PATH = "results/svhn/1vs7/"
#
# CHOOSEN_CLASSES = [1, 7]
# MAX_TRAIN_FRACTION = 1.0

# ####### FashionMNIST ########
# DATASET_NAME = "FashionMNIST"
# TITLE = "T-shirt/top (0) vs Dress (3)"
#
# DATA_PATH = "./data"
# RESULTS_PATH = "results/fashion_mnist/0vs3/"
#
# CHOOSEN_CLASSES = [0, 3]
# MAX_TRAIN_FRACTION = 1.0

##############################################################################################################
# CNN Config
##############################################################################################################

BATCH_SIZE = 128
EPOCH = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_CLASSES = len(CHOOSEN_CLASSES)
CNN_CONFIG = {"batch_size": BATCH_SIZE, "epoch": EPOCH, "device": DEVICE}

logging.basicConfig(level=logging.INFO,
                    format="%(message)s",
                    handlers=[logging.FileHandler(RESULTS_PATH + "logs.txt", mode='w'), logging.StreamHandler()])

logging.info("Are GPUs available? " + str(torch.cuda.is_available()) + "\n")

##############################################################################################################
# Data
##############################################################################################################

numpy_data = dict()
(numpy_data["train_images"], numpy_data["train_labels"]), (numpy_data["test_images"], numpy_data["test_labels"]) = \
    get_dataset(DATA_PATH, DATASET_NAME, is_numpy=True)

pytorch_data = dict()
pytorch_data["trainset"], pytorch_data["testset"] = get_dataset(DATA_PATH, DATASET_NAME, is_numpy=False)

fraction_of_train_samples_space = np.geomspace(0.01, MAX_TRAIN_FRACTION, num=10)

total_train_samples = sum([len(np.argwhere(numpy_data["train_labels"] == class_index))
                           for class_index in CHOOSEN_CLASSES])
number_of_train_samples_space = [int(i) for i in list(fraction_of_train_samples_space * total_train_samples)]

IMG_SHAPE = numpy_data["train_images"].shape[1:]

##############################################################################################################
# Helpers
##############################################################################################################


def print_items(fraction_of_train_samples, number_of_train_samples, best_accuracy, time_taken, cnn_model, cnn_config):
    if cnn_model:
        logging.info("CNN Config: " + str(cnn_config))
    logging.info("Train Fraction: " + str(fraction_of_train_samples))
    logging.info("# of Train Samples: " + str(number_of_train_samples))
    logging.info("Accuracy: " + str(best_accuracy))
    logging.info("Experiment Runtime: " + str(time_taken) + "\n")


def print_old_results(file_name, cnn_model, cnn_config):
    global fraction_of_train_samples_space, number_of_train_samples_space

    accuracy_scores = np.load(file_name)

    for fraction_of_train_samples, number_of_train_samples, (best_accuracy, time_taken) in zip(fraction_of_train_samples_space, number_of_train_samples_space, accuracy_scores):
        print_items(fraction_of_train_samples, number_of_train_samples, best_accuracy, time_taken, cnn_model, cnn_config)
        logging.info("")
    return accuracy_scores


def run_experiment(experiment, results_file_name, experiment_name, repeats=2, cnn_model=None, cnn_config={}):
    global fraction_of_train_samples_space, numpy_data, pytorch_data

    logging.info("##################################################################")
    logging.info("acc vs n_samples: " + experiment_name + "\n")

    acc_vs_n = list()
    file_name = RESULTS_PATH + results_file_name + ".npy"

    if not os.path.exists(file_name):
        for fraction_of_train_samples, number_of_train_samples in zip(fraction_of_train_samples_space, number_of_train_samples_space):

            if not cnn_model:
                start = time.time()
                best_accuracy = np.mean(
                    [experiment(DATASET_NAME, numpy_data, CHOOSEN_CLASSES, fraction_of_train_samples) for _ in range(repeats)])
                end = time.time()

            else:
                start = time.time()
                best_accuracy = np.mean(
                    [experiment(DATASET_NAME, cnn_model, pytorch_data, CHOOSEN_CLASSES, fraction_of_train_samples, cnn_config) for _ in range(repeats)])
                end = time.time()

            time_taken = (end - start)/float(repeats)
            acc_vs_n.append((best_accuracy, time_taken))

            print_items(fraction_of_train_samples, number_of_train_samples, best_accuracy, time_taken, cnn_model, cnn_config)

        np.save(file_name, acc_vs_n)

    else:
        acc_vs_n = print_old_results(file_name, cnn_model, cnn_config)

    logging.info("##################################################################")

    return acc_vs_n


##############################################################################################################
# Runners
##############################################################################################################

if __name__ == '__main__':

    script_start = time.time()

    # Naive RF
    run_experiment(run_naive_rf, "naive_rf_acc_vs_n", "Naive RF")

    # # Naive RerF
    # run_experiment(run_naive_rerf, "naive_rf_pyrerf_acc_vs_n", "Naive RF (pyrerf)")

    # DeepConvRF Unshared
    run_experiment(run_one_layer_deep_conv_rf_unshared, "deep_conv_rf_old_acc_vs_n", "DeepConvRF (1-layer, unshared)")
    run_experiment(run_two_layer_deep_conv_rf_unshared, "deep_conv_rf_old_two_layer_acc_vs_n", "DeepConvRF (2-layer, unshared)")

    # DeepConvRF Shared
    run_experiment(run_one_layer_deep_conv_rf_shared, "deep_conv_rf_acc_vs_n", "DeepConvRF (1-layer, shared)")
    run_experiment(run_two_layer_deep_conv_rf_shared, "deep_conv_rf_two_layer_acc_vs_n", "DeepConvRF (2-layer, shared)")

    # # DeepConvRerF Shared
    # run_experiment(run_one_layer_deep_conv_rerf_shared, "deep_conv_rf_pyrerf_acc_vs_n", "DeepConvRF (1-layer, shared, pyrerf)")
    # run_experiment(run_two_layer_deep_conv_rerf_shared, "deep_conv_rf_pyrerf_two_layer_acc_vs_n", "DeepConvRF (2-layer, shared, pyrerf)")

    # CNN
    cnn_acc_vs_n_config = copy.deepcopy(CNN_CONFIG)
    cnn_acc_vs_n_config.update({'model': 0, 'lr': 0.001, 'weight_decay': 1e-05})
    run_experiment(run_cnn, "cnn_acc_vs_n", "CNN (1-layer, 1-filter)", cnn_model=SimpleCNN1layer(1, NUM_CLASSES, IMG_SHAPE), cnn_config=cnn_acc_vs_n_config, repeats=3)

    cnn32_acc_vs_n_config = copy.deepcopy(CNN_CONFIG)
    cnn32_acc_vs_n_config.update({'model': 1, 'lr': 0.001, 'weight_decay': 0.1})
    run_experiment(run_cnn, "cnn32_acc_vs_n", "CNN (1-layer, 32-filter)", cnn_model=SimpleCNN1layer(32, NUM_CLASSES, IMG_SHAPE), cnn_config=cnn32_acc_vs_n_config, repeats=3)

    cnn32_two_layer_acc_vs_n_config = copy.deepcopy(CNN_CONFIG)
    cnn32_two_layer_acc_vs_n_config.update({'model': 2, 'lr': 0.001, 'weight_decay': 0.1})
    run_experiment(run_cnn, "cnn32_two_layer_acc_vs_n", "CNN (2-layer, 32-filter)", cnn_model=SimpleCNN2Layers(32, NUM_CLASSES, IMG_SHAPE), cnn_config=cnn32_two_layer_acc_vs_n_config, repeats=3)

    # Best CNN
    run_experiment(run_cnn, "cnn_best_acc_vs_n", "CNN (ResNet18)", cnn_model=ResNet18(NUM_CLASSES, IMG_SHAPE), cnn_config=CNN_CONFIG, repeats=3)

    script_end = time.time()
    logging.info("Total Runtime: " + str(script_end - script_start) + "\n")
