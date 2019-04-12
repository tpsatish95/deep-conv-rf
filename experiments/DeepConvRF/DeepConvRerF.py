# general imports
import os.path
import warnings
import time

import torchvision.datasets as datasets

import numpy as np
from multiprocessing import cpu_count
from sklearn.metrics import accuracy_score

from RerF import fastRerF, fastPredict

warnings.filterwarnings("ignore")


##########
# Settings
##########
base_path = ""
# base_path = "rerf/1vs9/"
cifar_data_path = "./data"

class_one = 1
class_two = 9

fraction_of_train_samples_space = np.geomspace(0.01, 1.0, num=10)

NUM_TREES = 1000
TREE_TYPE = "binnedBase"


###########################################################################################################
# Data Preparation
###########################################################################################################

def normalize(x):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    return (x - mean) / std


# train data
cifar_trainset = datasets.CIFAR10(root=cifar_data_path, train=True, download=True, transform=None)
cifar_train_images = normalize(cifar_trainset.data)
cifar_train_labels = np.array(cifar_trainset.targets)

# test data
cifar_testset = datasets.CIFAR10(root=cifar_data_path, train=False, download=True, transform=None)
cifar_test_images = normalize(cifar_testset.data)
cifar_test_labels = np.array(cifar_testset.targets)


class ConvRF(object):
    def __init__(self, kernel_size=5, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.kernel_forest = None
        self.num_trees = NUM_TREES

    def _convolve_chop(self, images, labels=None, flatten=False):

        batch_size, in_dim, _, num_channels = images.shape

        out_dim = int((in_dim - self.kernel_size) / self.stride) + 1  # calculate output dimensions

        # create matrix to hold the chopped images
        out_images = np.zeros((batch_size, out_dim, out_dim,
                               self.kernel_size, self.kernel_size, num_channels))
        out_labels = None

        curr_y = out_y = 0
        # move kernel vertically across the image
        while curr_y + self.kernel_size <= in_dim:
            curr_x = out_x = 0
            # move kernel horizontally across the image
            while curr_x + self.kernel_size <= in_dim:
                # chop images
                out_images[:, out_x, out_y] = images[:, curr_x:curr_x +
                                                     self.kernel_size, curr_y:curr_y+self.kernel_size, :]
                curr_x += self.stride
                out_x += 1
            curr_y += self.stride
            out_y += 1

        if flatten:
            out_images = out_images.reshape(batch_size, out_dim, out_dim, -1)

        if labels is not None:
            out_labels = np.zeros((batch_size, out_dim, out_dim))
            out_labels[:, ] = labels.reshape(-1, 1, 1)

        return out_images, out_labels

    def rf_predict(self, sample):
        return np.array(self.kernel_forest.predict_post(sample.tolist())[1] / float(self.num_trees))

    def convolve_fit(self, images, labels):
        sub_images, sub_labels = self._convolve_chop(images, labels=labels, flatten=True)

        batch_size, out_dim, _, _ = sub_images.shape

        all_sub_images = sub_images.reshape(batch_size*out_dim*out_dim, -1)
        all_sub_labels = sub_labels.reshape(batch_size*out_dim*out_dim, -1)

        self.kernel_forest = fastRerF(X=all_sub_images,
                                      Y=all_sub_labels,
                                      forestType=TREE_TYPE,
                                      trees=self.num_trees,
                                      numCores=cpu_count() - 1)

        convolved_image = np.zeros((images.shape[0], out_dim, out_dim, 1))
        for i in range(out_dim):
            for j in range(out_dim):
                convolved_image[:, i, j] = np.array([self.rf_predict(sample) for sample in sub_images[:, i, j]])[..., np.newaxis]
        return convolved_image

    def convolve_predict(self, images):
        if not self.kernel_forest:
            raise Exception("Should fit training data before predicting")

        sub_images, _ = self._convolve_chop(images, flatten=True)

        batch_size, out_dim, _, _ = sub_images.shape

        kernel_predictions = np.zeros((images.shape[0], out_dim, out_dim, 1))

        for i in range(out_dim):
            for j in range(out_dim):
                kernel_predictions[:, i, j] = np.array([self.rf_predict(sample) for sample in sub_images[:, i, j]])[..., np.newaxis]
        return kernel_predictions


def run_naive_rf(train_images, train_labels, test_images, test_labels, fraction_of_train_samples, class1=3, class2=5):
    num_train_samples_class_1 = int(np.sum(train_labels == class1) * fraction_of_train_samples)
    num_train_samples_class_2 = int(np.sum(train_labels == class2) * fraction_of_train_samples)

    # get only train images and labels for class 1 and class 2
    train_images = np.concatenate([train_images[train_labels == class1][:num_train_samples_class_1],
                                   train_images[train_labels == class2][:num_train_samples_class_2]])
    train_labels = np.concatenate(
        [np.repeat(0, num_train_samples_class_1), np.repeat(1, num_train_samples_class_2)])

    # get only test images and labels for class 1 and class 2
    test_images = np.concatenate([test_images[test_labels == class1],
                                  test_images[test_labels == class2]])
    test_labels = np.concatenate(
        [np.repeat(0, np.sum(test_labels == class1)), np.repeat(1, np.sum(test_labels == class2))])

    # Train
    forest = fastRerF(X=train_images.reshape(-1, 32*32*3),
                      Y=train_labels,
                      forestType=TREE_TYPE,
                      trees=100,
                      numCores=cpu_count() - 1)
    # forest.printParameters()

    # Test
    test_preds = fastPredict(test_images.reshape(-1, 32*32*3), forest)
    return accuracy_score(test_labels, test_preds)


def run_one_layer_deep_conv_rf(train_images, train_labels, test_images, test_labels, fraction_of_train_samples, class1=3, class2=5):
    num_train_samples_class_1 = int(np.sum(train_labels == class1) * fraction_of_train_samples)
    num_train_samples_class_2 = int(np.sum(train_labels == class2) * fraction_of_train_samples)

    # get only train images and labels for class 1 and class 2
    train_images = np.concatenate([train_images[train_labels == class1][:num_train_samples_class_1],
                                   train_images[train_labels == class2][:num_train_samples_class_2]])
    train_labels = np.concatenate(
        [np.repeat(0, num_train_samples_class_1), np.repeat(1, num_train_samples_class_2)])

    # get only test images and labels for class 1 and class 2
    test_images = np.concatenate([test_images[test_labels == class1],
                                  test_images[test_labels == class2]])
    test_labels = np.concatenate(
        [np.repeat(0, np.sum(test_labels == class1)), np.repeat(1, np.sum(test_labels == class2))])

    # Train
    # ConvRF (layer 1)
    conv1 = ConvRF(kernel_size=10, stride=2)
    conv1_map = conv1.convolve_fit(train_images, train_labels)

    # Full RF
    conv1_full_RF = fastRerF(X=conv1_map.reshape(len(train_images), -1),
                             Y=train_labels,
                             forestType=TREE_TYPE,
                             trees=100,
                             numCores=cpu_count() - 1)

    # Test (after ConvRF 1 and Full RF)
    conv1_map_test = conv1.convolve_predict(test_images)
    test_preds = fastPredict(conv1_map_test.reshape(len(test_images), -1), conv1_full_RF)

    return accuracy_score(test_labels, test_preds)


def run_two_layer_deep_conv_rf(train_images, train_labels, test_images, test_labels, fraction_of_train_samples, class1=3, class2=5):
    num_train_samples_class_1 = int(np.sum(train_labels == class1) * fraction_of_train_samples)
    num_train_samples_class_2 = int(np.sum(train_labels == class2) * fraction_of_train_samples)

    # get only train images and labels for class 1 and class 2
    train_images = np.concatenate([train_images[train_labels == class1][:num_train_samples_class_1],
                                   train_images[train_labels == class2][:num_train_samples_class_2]])
    train_labels = np.concatenate(
        [np.repeat(0, num_train_samples_class_1), np.repeat(1, num_train_samples_class_2)])

    # get only test images and labels for class 1 and class 2
    test_images = np.concatenate([test_images[test_labels == class1],
                                  test_images[test_labels == class2]])
    test_labels = np.concatenate(
        [np.repeat(0, np.sum(test_labels == class1)), np.repeat(1, np.sum(test_labels == class2))])

    # Train
    # ConvRF (layer 1)
    conv1 = ConvRF(kernel_size=10, stride=2)
    conv1_map = conv1.convolve_fit(train_images, train_labels)

    # ConvRF (layer 2)
    conv2 = ConvRF(kernel_size=7, stride=1)
    conv2_map = conv2.convolve_fit(conv1_map, train_labels)

    # Full RF
    conv1_full_RF = fastRerF(X=conv2_map.reshape(len(train_images), -1),
                             Y=train_labels,
                             forestType=TREE_TYPE,
                             trees=100,
                             numCores=cpu_count() - 1)

    # Test (after ConvRF 2 and Full RF)
    conv1_map_test = conv1.convolve_predict(test_images)
    conv2_map_test = conv2.convolve_predict(conv1_map_test)
    test_preds = fastPredict(conv2_map_test.reshape(len(test_images), -1), conv1_full_RF)

    return accuracy_score(test_labels, test_preds)


###############################################################################
# Experiments
###############################################################################

def print_old_results(file_name):
    global fraction_of_train_samples_space
    accuracy_scores = np.load(file_name)
    for fraction_of_train_samples, (best_accuracy, time_taken) in zip(fraction_of_train_samples_space, accuracy_scores):
        print("Train Fraction:", str(fraction_of_train_samples))
        print("Accuracy:", str(best_accuracy))
        print("Experiment Runtime: " + str(time_taken), "\n")
        print()
    return accuracy_scores


def run_experiment(experiment, experiment_result_file, text, cnn_model=None, class1=class_one, class2=class_two):
    global fraction_of_train_samples_space
    repeats = 2

    print("##################################################################")
    print("acc vs n_samples: " + text + "\n")
    acc_vs_n = list()
    file_name = base_path+experiment_result_file+".npy"
    if not os.path.exists(file_name):
        for fraction_of_train_samples in fraction_of_train_samples_space:
            if not cnn_model:
                start = time.time()
                best_accuracy = np.mean([experiment(cifar_train_images, cifar_train_labels, cifar_test_images,
                                                    cifar_test_labels, fraction_of_train_samples, class1, class2) for _ in range(repeats)])
                end = time.time()
            else:
                start = time.time()
                best_accuracy = np.mean([experiment(cnn_model, cifar_train_images, cifar_train_labels, cifar_test_images,
                                                    cifar_test_labels, fraction_of_train_samples, class1, class2) for _ in range(repeats)])
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


if __name__ == '__main__':
    naive_rf_pyrerf_acc_vs_n, naive_rf_pyrerf_acc_vs_n_times = list(zip(*run_experiment(run_naive_rf, "naive_rf_pyrerf_acc_vs_n", "Naive RF (pyrerf)")))
    deep_conv_rf_pyrerf_acc_vs_n, deep_conv_rf_pyrerf_acc_vs_n_times = list(zip(*run_experiment(run_one_layer_deep_conv_rf, "deep_conv_rf_pyrerf_acc_vs_n", "DeepConvRF (1-layer, shared, pyrerf)")))
    # deep_conv_rf_pyrerf_two_layer_acc_vs_n, deep_conv_rf_pyrerf_two_layer_acc_vs_n_times = list(zip(*run_experiment(run_two_layer_deep_conv_rf, "deep_conv_rf_pyrerf_two_layer_acc_vs_n", "DeepConvRF (2-layer, shared, pyrerf)")))
