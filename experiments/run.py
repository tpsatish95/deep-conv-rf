import copy
import logging
import os
import sys
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

'''General Settings'''
DATA_PATH = "./data"
MIN_TRAIN_SAMPLES = 10
MAX_TRAIN_SAMPLES = 100
N_TRIALS = 10

RUN_RF = True
RUN_CNN = True

if len(sys.argv) > 1:
    DATASET_NAME = str(sys.argv[1])
    CHOOSEN_CLASSES = [int(i) for i in sys.argv[2:]]
else:
    # DATASET_NAME = "CIFAR10"
    DATASET_NAME = "SVHN"
    # DATASET_NAME = "FashionMNIST"

    # CHOOSEN_CLASSES = [3, 8]


##############################################################################################################

if DATASET_NAME == "CIFAR10":
    '''CIFAR10'''
    CIFAR10_MAP = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
                   4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    TITLE = " vs ".join([CIFAR10_MAP[i].capitalize() +
                         " (" + str(i) + ")" for i in CHOOSEN_CLASSES])
    RESULTS_PATH = "results/cifar10/" + "vs".join([str(i) for i in CHOOSEN_CLASSES]) + "/" + str(MIN_TRAIN_SAMPLES) + "_to_" + str(MAX_TRAIN_SAMPLES) + "/"
elif DATASET_NAME == "SVHN":
    '''SVHN'''
    TITLE = " vs ".join([str(i) for i in CHOOSEN_CLASSES])
    RESULTS_PATH = "results/svhn/" + "vs".join([str(i) for i in CHOOSEN_CLASSES]) + "/" + str(MIN_TRAIN_SAMPLES) + "_to_" + str(MAX_TRAIN_SAMPLES) + "/"
elif DATASET_NAME == "FashionMNIST":
    '''FashionMNIST'''
    FashionMNIST_MAP = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                        'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    TITLE = " vs ".join([FashionMNIST_MAP[i] + " (" + str(i) + ")" for i in CHOOSEN_CLASSES])
    RESULTS_PATH = "results/fashion_mnist/" + "vs".join([str(i) for i in CHOOSEN_CLASSES]) + "/" + str(MIN_TRAIN_SAMPLES) + "_to_" + str(MAX_TRAIN_SAMPLES) + "/"

# create the results directory
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

##############################################################################################################
# CNN Config
##############################################################################################################

BATCH_SIZE = 8
EPOCH = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_CLASSES = len(CHOOSEN_CLASSES)
CNN_CONFIG = {"batch_size": BATCH_SIZE, "epoch": EPOCH, "device": DEVICE}

if __name__ == '__main__':
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
pytorch_data["trainset"], pytorch_data["testset"] = get_dataset(
    DATA_PATH, DATASET_NAME, is_numpy=False)

class_wise_train_indices = [np.argwhere(
    numpy_data["train_labels"] == class_index).flatten() for class_index in CHOOSEN_CLASSES]

total_train_samples = sum([len(ci) for ci in class_wise_train_indices])

MIN_TRAIN_FRACTION = MIN_TRAIN_SAMPLES / total_train_samples
MAX_TRAIN_FRACTION = MAX_TRAIN_SAMPLES / total_train_samples
fraction_of_train_samples_space = np.geomspace(MIN_TRAIN_FRACTION, MAX_TRAIN_FRACTION, num=10)

number_of_train_samples_space = [int(i) for i in list(fraction_of_train_samples_space * total_train_samples)]

train_indices_all_trials = list()
for n in range(N_TRIALS):
    train_indices = list()
    for frac in fraction_of_train_samples_space:
        sub_sample = np.concatenate([np.random.choice(class_indices, int(
            len(class_indices)*frac), replace=False) for class_indices in class_wise_train_indices]).flatten()
        train_indices.append(sub_sample)
    train_indices_all_trials.append(train_indices)


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

    accuracy_scores = [[np.mean(list(zip(*i))[0]), np.mean(list(zip(*i))[1])]
                       for i in list(zip(*np.load(file_name)))]

    for fraction_of_train_samples, number_of_train_samples, (best_accuracy, time_taken) in zip(fraction_of_train_samples_space, number_of_train_samples_space, accuracy_scores):
        print_items(fraction_of_train_samples, number_of_train_samples,
                    best_accuracy, time_taken, cnn_model, cnn_config)
        logging.info("")
    return accuracy_scores


def run_experiment(experiment, results_file_name, experiment_name, cnn_model=None, cnn_config={}):
    global fraction_of_train_samples_space, numpy_data, pytorch_data, train_indices_all_trials

    logging.info("##################################################################")
    logging.info("acc vs n_samples: " + experiment_name + "\n")

    acc_vs_n_all_trials = list()
    file_name = RESULTS_PATH + results_file_name + ".npy"

    if not os.path.exists(file_name):
        for trial_number, train_indices in zip(range(N_TRIALS), train_indices_all_trials):
            logging.info("Trial " + str(trial_number+1) + "\n")
            acc_vs_n = list()
            for fraction_of_train_samples, sub_train_indices in zip(fraction_of_train_samples_space, train_indices):
                if not cnn_model:
                    start = time.time()
                    accuracy = experiment(DATASET_NAME, numpy_data,
                                          CHOOSEN_CLASSES, sub_train_indices)
                    end = time.time()
                else:
                    start = time.time()
                    accuracy = experiment(DATASET_NAME, cnn_model, pytorch_data,
                                          CHOOSEN_CLASSES, sub_train_indices, cnn_config)
                    end = time.time()
                time_taken = (end - start)

                print_items(fraction_of_train_samples, len(sub_train_indices),
                            accuracy, time_taken, cnn_model, cnn_config)

                acc_vs_n.append((accuracy, time_taken))

            acc_vs_n_all_trials.append(acc_vs_n)

        np.save(file_name, acc_vs_n_all_trials)

    else:
        acc_vs_n = print_old_results(file_name, cnn_model, cnn_config)

    logging.info("##################################################################")

    return acc_vs_n


##############################################################################################################
# Runners
##############################################################################################################

if __name__ == '__main__':

    script_start = time.time()

    if RUN_RF:
        # Naive RF
        run_experiment(run_naive_rf,
                       "naive_rf_acc_vs_n", "Naive RF")

        # # Naive RerF
        # run_experiment(run_naive_rerf, "naive_rf_pyrerf_acc_vs_n", "Naive RF (pyrerf)")

        # DeepConvRF Unshared
        run_experiment(run_one_layer_deep_conv_rf_unshared,
                       "deep_conv_rf_old_acc_vs_n", "DeepConvRF (1-layer, unshared)")
        run_experiment(run_two_layer_deep_conv_rf_unshared,
                       "deep_conv_rf_old_two_layer_acc_vs_n", "DeepConvRF (2-layer, unshared)")

        # DeepConvRF Shared
        run_experiment(run_one_layer_deep_conv_rf_shared,
                       "deep_conv_rf_acc_vs_n", "DeepConvRF (1-layer, shared)")
        run_experiment(run_two_layer_deep_conv_rf_shared,
                       "deep_conv_rf_two_layer_acc_vs_n", "DeepConvRF (2-layer, shared)")

        # # DeepConvRerF Shared
        # run_experiment(run_one_layer_deep_conv_rerf_shared, "deep_conv_rf_pyrerf_acc_vs_n", "DeepConvRF (1-layer, shared, pyrerf)")
        # run_experiment(run_two_layer_deep_conv_rerf_shared, "deep_conv_rf_pyrerf_two_layer_acc_vs_n", "DeepConvRF (2-layer, shared, pyrerf)")

    if RUN_CNN:
        # CNN
        cnn_acc_vs_n_config = copy.deepcopy(CNN_CONFIG)
        cnn_acc_vs_n_config.update({'model': 0, 'lr': 0.001, 'weight_decay': 1e-05})
        run_experiment(run_cnn, "cnn_acc_vs_n", "CNN (1-layer, 1-filter)",
                       cnn_model=SimpleCNN1layer(1, NUM_CLASSES, IMG_SHAPE), cnn_config=cnn_acc_vs_n_config)

        cnn32_acc_vs_n_config = copy.deepcopy(CNN_CONFIG)
        cnn32_acc_vs_n_config.update({'model': 1, 'lr': 0.001, 'weight_decay': 0.1})
        run_experiment(run_cnn, "cnn32_acc_vs_n", "CNN (1-layer, 32-filter)",
                       cnn_model=SimpleCNN1layer(32, NUM_CLASSES, IMG_SHAPE), cnn_config=cnn32_acc_vs_n_config)

        cnn32_two_layer_acc_vs_n_config = copy.deepcopy(CNN_CONFIG)
        cnn32_two_layer_acc_vs_n_config.update({'model': 2, 'lr': 0.001, 'weight_decay': 0.1})
        run_experiment(run_cnn, "cnn32_two_layer_acc_vs_n", "CNN (2-layer, 32-filter)",
                       cnn_model=SimpleCNN2Layers(32, NUM_CLASSES, IMG_SHAPE), cnn_config=cnn32_two_layer_acc_vs_n_config)

        # Best CNN
        run_experiment(run_cnn, "cnn_best_acc_vs_n", "CNN (ResNet18)",
                       cnn_model=ResNet18(NUM_CLASSES, IMG_SHAPE), cnn_config=CNN_CONFIG)

    script_end = time.time()
    logging.info("Total Runtime: " + str(script_end - script_start) + "\n")
