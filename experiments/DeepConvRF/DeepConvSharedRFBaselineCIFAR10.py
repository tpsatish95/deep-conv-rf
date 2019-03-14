# general imports
import seaborn as sns
import matplotlib
import torchvision.datasets as datasets
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os.path

# import sys
# sys.stdout = open("deep_conv_rf_logs.txt", "w+")


import warnings
warnings.filterwarnings("ignore")


scale = np.mean(np.arange(0, 256))


def normalize(x): return (x - scale) / scale


# train data
cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
cifar_train_images = normalize(cifar_trainset.data)
cifar_train_labels = np.array(cifar_trainset.targets)

# get only train images and labels for two classes: 3 (cat) and 5 (dog)
cifar_train_images_3_5 = np.concatenate(
    [cifar_train_images[cifar_train_labels == 3], cifar_train_images[cifar_train_labels == 5]])
cifar_train_labels_3_5 = np.concatenate(
    [np.repeat(0, np.sum(cifar_train_labels == 3)), np.repeat(1, np.sum(cifar_train_labels == 5))])

# test data
cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
cifar_test_images = normalize(cifar_testset.data)
cifar_test_labels = np.array(cifar_testset.targets)

# get only test images and labels for two classes: 3 (cat) and 5 (dog)
cifar_test_images_3_5 = np.concatenate(
    [cifar_test_images[cifar_test_labels == 3], cifar_test_images[cifar_test_labels == 5]])
cifar_test_labels_3_5 = np.concatenate(
    [np.repeat(0, np.sum(cifar_test_labels == 3)), np.repeat(1, np.sum(cifar_test_labels == 5))])
# print(np.min(cifar_train_images_3_5))
# print(np.max(cifar_train_images_3_5))


# # All of CIFAR 10
# print(cifar_train_images.shape)
# print(cifar_test_images.shape)
# print(np.unique(cifar_test_labels))
#
# # Cats vs Dogs (CIFAR)
# print(cifar_train_images_3_5.shape)
# print(cifar_test_images_3_5.shape)
# print(np.unique(cifar_test_labels_3_5))
#
# print("Naive Random Forest (by flattening the entire image)")
# # Train
# clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
# clf.fit(cifar_train_images_3_5.reshape(-1, 32*32*3), cifar_train_labels_3_5)
# # Test
# cifar_test_preds_3_5 = clf.predict(cifar_test_images_3_5.reshape(-1, 32*32*3))
# print("Test Accuracy: " + str(accuracy_score(cifar_test_labels_3_5, cifar_test_preds_3_5)))
# print("Validation Confusion Matrix: \n" +
#       str(confusion_matrix(cifar_test_labels_3_5, cifar_test_preds_3_5)))


class ConvRFOld(object):
    def __init__(self, kernel_size=5, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.kernel_forests = None

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

    def convolve_fit(self, images, labels):
        num_channels = images.shape[-1]
        sub_images, sub_labels = self._convolve_chop(images, labels=labels, flatten=True)

        batch_size, out_dim, _, _ = sub_images.shape
        self.kernel_forests = np.zeros((out_dim, out_dim), dtype=np.int).tolist()
        convolved_image = np.zeros((images.shape[0], out_dim, out_dim, 1))

        for i in range(out_dim):
            for j in range(out_dim):
                self.kernel_forests[i][j] = RandomForestClassifier(n_estimators=32)
                self.kernel_forests[i][j].fit(sub_images[:, i, j], sub_labels[:, i, j])
                convolved_image[:, i, j] = self.kernel_forests[i][j].predict_proba(
                    sub_images[:, i, j])[..., 1][..., np.newaxis]
        return convolved_image

    def convolve_predict(self, images):
        if not self.kernel_forests:
            raise Exception("Should fit training data before predicting")

        num_channels = images.shape[-1]
        sub_images, _ = self._convolve_chop(images, flatten=True)

        batch_size, out_dim, _, _ = sub_images.shape

        kernel_predictions = np.zeros((images.shape[0], out_dim, out_dim, 1))

        for i in range(out_dim):
            for j in range(out_dim):
                kernel_predictions[:, i, j] = self.kernel_forests[i][j].predict_proba(
                    sub_images[:, i, j])[..., 1][..., np.newaxis]
        return kernel_predictions


class ConvRF(object):
    def __init__(self, kernel_size=5, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.kernel_forest = None

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

    def convolve_fit(self, images, labels):
        num_channels = images.shape[-1]
        sub_images, sub_labels = self._convolve_chop(images, labels=labels, flatten=True)

        batch_size, out_dim, _, _ = sub_images.shape

        all_sub_images = sub_images.reshape(batch_size*out_dim*out_dim, -1)
        all_sub_labels = sub_labels.reshape(batch_size*out_dim*out_dim, -1)

        # print(all_sub_images.shape)

        # fit all sub images to a forest
#         self.kernel_forest = RandomForestClassifier(n_estimators=500, n_jobs=-1, warm_start=True)
        self.kernel_forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
        self.kernel_forest.fit(all_sub_images, all_sub_labels)

        convolved_image = np.zeros((images.shape[0], out_dim, out_dim, 1))

        for i in range(out_dim):
            for j in range(out_dim):
                #                 self.kernel_forest.fit(sub_images[:, i, j], sub_labels[:, i, j])
                convolved_image[:, i, j] = self.kernel_forest.predict_proba(
                    sub_images[:, i, j])[..., 1][..., np.newaxis]
        return convolved_image

    def convolve_predict(self, images):
        if not self.kernel_forest:
            raise Exception("Should fit training data before predicting")

        num_channels = images.shape[-1]
        sub_images, _ = self._convolve_chop(images, flatten=True)

        batch_size, out_dim, _, _ = sub_images.shape

        kernel_predictions = np.zeros((images.shape[0], out_dim, out_dim, 1))

        for i in range(out_dim):
            for j in range(out_dim):
                kernel_predictions[:, i, j] = self.kernel_forest.predict_proba(
                    sub_images[:, i, j])[..., 1][..., np.newaxis]
        return kernel_predictions


def run_naive_rf(train_images, train_labels, test_images, test_labels, fraction_of_train_samples, class1=3, class2=8):
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
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    clf.fit(train_images.reshape(-1, 32*32*3), train_labels)
    # Test
    test_preds = clf.predict(test_images.reshape(-1, 32*32*3))
    return accuracy_score(test_labels, test_preds)


def run_one_layer_deep_conv_rf_old(train_images, train_labels, test_images, test_labels, fraction_of_train_samples, class1=3, class2=8):
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
    conv1 = ConvRFOld(kernel_size=10, stride=2)
    conv1_map = conv1.convolve_fit(train_images, train_labels)

    # Full RF
    conv1_full_RF = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    conv1_full_RF.fit(conv1_map.reshape(len(train_images), -1), train_labels)

    # Test (after ConvRF 1 and Full RF)
    conv1_map_test = conv1.convolve_predict(test_images)
    mnist_test_preds = conv1_full_RF.predict(conv1_map_test.reshape(len(test_images), -1))

    return accuracy_score(test_labels, mnist_test_preds)


def run_one_layer_deep_conv_rf(train_images, train_labels, test_images, test_labels, fraction_of_train_samples, class1=3, class2=8):
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
    conv1_full_RF = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    conv1_full_RF.fit(conv1_map.reshape(len(train_images), -1), train_labels)

    # Test (after ConvRF 1 and Full RF)
    conv1_map_test = conv1.convolve_predict(test_images)
    mnist_test_preds = conv1_full_RF.predict(conv1_map_test.reshape(len(test_images), -1))

    return accuracy_score(test_labels, mnist_test_preds)


fraction_of_train_samples_space = np.geomspace(0.01, 0.1, num=10)


def print_old_results(file_name):
    global fraction_of_train_samples_space
    accuracy_scores = np.load(file_name)
    for fraction_of_train_samples, best_accuracy in zip(fraction_of_train_samples_space, accuracy_scores):
        print("Train Fraction:", str(fraction_of_train_samples))
        print("Accuracy:", str(best_accuracy))
        print()
    return accuracy_scores


print("acc vs n_samples: Naive RF\n")
naive_rf_acc_vs_n = list()
file_name = "naive_rf_acc_vs_n.npy"
if not os.path.exists(file_name):
    for fraction_of_train_samples in fraction_of_train_samples_space:
        best_accuracy = np.mean([run_naive_rf(cifar_train_images, cifar_train_labels, cifar_test_images,
                                              cifar_test_labels, fraction_of_train_samples, 3, 5) for _ in range(2)])
        naive_rf_acc_vs_n.append(best_accuracy)
        print("Train Fraction:", str(fraction_of_train_samples))
        print("Accuracy:", str(best_accuracy))
        print()
    np.save(file_name, naive_rf_acc_vs_n)
else:
    naive_rf_acc_vs_n = print_old_results(file_name)
print()


print("acc vs n_samples: DeepConvRF (unshared)\n")
deep_conv_rf_old_acc_vs_n = list()
file_name = "deep_conv_rf_old_acc_vs_n.npy"
if not os.path.exists(file_name):
    for fraction_of_train_samples in fraction_of_train_samples_space:
        best_accuracy = np.mean([run_one_layer_deep_conv_rf_old(cifar_train_images, cifar_train_labels,
                                                                cifar_test_images, cifar_test_labels, fraction_of_train_samples, 3, 5) for _ in range(2)])
        deep_conv_rf_old_acc_vs_n.append(best_accuracy)
        print("Train Fraction:", str(fraction_of_train_samples))
        print("Accuracy:", str(best_accuracy))
        print()
    np.save(file_name, deep_conv_rf_old_acc_vs_n)
else:
    deep_conv_rf_old_acc_vs_n = print_old_results(file_name)
print()


print("acc vs n_samples: DeepConvRF (shared)\n")
deep_conv_rf_acc_vs_n = list()
file_name = "deep_conv_rf_acc_vs_n.npy"
if not os.path.exists(file_name):
    for fraction_of_train_samples in fraction_of_train_samples_space:
        best_accuracy = np.mean([run_one_layer_deep_conv_rf(cifar_train_images, cifar_train_labels,
                                                            cifar_test_images, cifar_test_labels, fraction_of_train_samples, 3, 5) for _ in range(2)])
        deep_conv_rf_acc_vs_n.append(best_accuracy)
        print("Train Fraction:", str(fraction_of_train_samples))
        print("Accuracy:", str(best_accuracy))
        print()
    np.save(file_name, deep_conv_rf_acc_vs_n)
else:
    deep_conv_rf_acc_vs_n = print_old_results(file_name)
print()


# plot
sns.set()

plt.rcParams['figure.figsize'] = 10, 5
plt.rcParams["legend.loc"] = "best"
plt.rcParams['figure.facecolor'] = 'white'

fig, ax = plt.subplots()  # create a new figure with a default 111 subplot
ax.plot(fraction_of_train_samples_space, naive_rf_acc_vs_n, marker='X', markerfacecolor='red',
        markersize=6, color='green', linewidth=3, linestyle=":", label="Naive RF")
ax.plot(fraction_of_train_samples_space, deep_conv_rf_old_acc_vs_n, marker='X', markerfacecolor='red',
        markersize=6, color='green', linewidth=3, linestyle="-.", label="Deep Conv RF (unshared)")
ax.plot(fraction_of_train_samples_space, deep_conv_rf_acc_vs_n, marker='X',
        markerfacecolor='red', markersize=6, color='green', linewidth=3, label="Deep Conv RF (shared)")

ax.set_xlabel('Fraction of Train Samples', fontsize=18)
ax.set_xscale('log')
ax.set_xticks(list(np.geomspace(0.01, 0.1, num=10)))
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax.set_ylabel('Accuracy on Test Set', fontsize=18)
# ax.set_ylim(0.85, 1)

ax.set_title("3 (cats) vs 5 (dogs) Classification", fontsize=18)
plt.legend()
plt.savefig("shared_deepconvrf_comparison.png")
