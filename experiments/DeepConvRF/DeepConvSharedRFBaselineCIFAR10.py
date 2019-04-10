# general imports
import sys
import os.path
import warnings
import time
import copy

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sns.set()
plt.style.use('seaborn')
warnings.filterwarnings("ignore")
# sys.stdout = open("deep_conv_rf_logs.txt", "w+")

##########
# Settings
##########
base_path = ""
base_path = "experiments/DeepConvRF/35_percent_data/1vs9/"

cifar_data_path = "./data"

class_one = 1
class_two = 9

MAX_TRAIN_FRACTION = 0.35


###########################################################################################################
# Data Preparation
###########################################################################################################

def normalize(x):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    return (x - mean) / std


# train data
cifar_trainset = datasets.CIFAR10(root=cifar_data_path, train=True, download=True, transform=None)
cifar_train_images = normalize(cifar_trainset.train_data)
cifar_train_labels = np.array(cifar_trainset.train_labels)

# test data
cifar_testset = datasets.CIFAR10(root=cifar_data_path, train=False, download=True, transform=None)
cifar_test_images = normalize(cifar_testset.test_data)
cifar_test_labels = np.array(cifar_testset.test_labels)

# # 3 (cat) vs 5 (dog) classification
#
# # get only train images and labels for two classes: 3 and 5
# cifar_train_images_3_5 = np.concatenate([cifar_train_images[cifar_train_labels==3], cifar_train_images[cifar_train_labels==5]])
# cifar_train_labels_3_5 = np.concatenate([np.repeat(0, np.sum(cifar_train_labels==3)), np.repeat(1, np.sum(cifar_train_labels==5))])
#
# # visualize data and labels
#
# # 3 (label 0) - cat
# index = 2500
# print("Label:", cifar_train_labels_3_5[index])
# plt.imshow(cifar_train_images_3_5[index])
# plt.show()
#
# # 5 (label 1) - dog
# index = 7500
# print("Label:", cifar_train_labels_3_5[index])
# plt.imshow(cifar_train_images_3_5[index])
# plt.show()


###########################################################################################################
# Different RandomForest Architectures
###########################################################################################################


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
        sub_images, sub_labels = self._convolve_chop(images, labels=labels, flatten=True)

        batch_size, out_dim, _, _ = sub_images.shape

        all_sub_images = sub_images.reshape(batch_size*out_dim*out_dim, -1)
        all_sub_labels = sub_labels.reshape(batch_size*out_dim*out_dim, -1)

        self.kernel_forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
        self.kernel_forest.fit(all_sub_images, all_sub_labels)

        convolved_image = np.zeros((images.shape[0], out_dim, out_dim, 1))
        for i in range(out_dim):
            for j in range(out_dim):
                convolved_image[:, i, j] = self.kernel_forest.predict_proba(
                    sub_images[:, i, j])[..., 1][..., np.newaxis]
        return convolved_image

    def convolve_predict(self, images):
        if not self.kernel_forest:
            raise Exception("Should fit training data before predicting")

        sub_images, _ = self._convolve_chop(images, flatten=True)

        batch_size, out_dim, _, _ = sub_images.shape

        kernel_predictions = np.zeros((images.shape[0], out_dim, out_dim, 1))

        for i in range(out_dim):
            for j in range(out_dim):
                kernel_predictions[:, i, j] = self.kernel_forest.predict_proba(
                    sub_images[:, i, j])[..., 1][..., np.newaxis]
        return kernel_predictions


###############################################################################
# Different CNN Architectures
###############################################################################


class SimpleCNN1layer(nn.Module):

    def __init__(self, num_filters, num_classes):
        super(SimpleCNN1layer, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=10, stride=2)
        self.fc1 = nn.Linear(144*num_filters, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        x = F.log_softmax(self.fc1(x), dim=1)
        return(x)


class SimpleCNN2Layers(nn.Module):

    def __init__(self, num_filters, num_classes):
        super(SimpleCNN2Layers, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=10, stride=2)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=7, stride=1)
        self.fc1 = nn.Linear(36*num_filters, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        x = F.log_softmax(self.fc1(x), dim=1)
        return(x)


# class BestCNN(torch.nn.Module):
#     def __init__(self):
#         super(BestCNN, self).__init__()
#         self.conv1 = torch.nn.Conv2d(3, 6, 5)
#         self.pool = torch.nn.MaxPool2d(2, 2)
#         self.conv2 = torch.nn.Conv2d(6, 16, 5)
#         self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = torch.nn.Linear(120, 84)
#         self.fc3 = torch.nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

###############################################################################
# Wrapper Code for Different Random Forest Experiments
###############################################################################


def run_naive_rf(train_images, train_labels, test_images, test_labels, fraction_of_train_samples, class1, class2):
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


def run_one_layer_deep_conv_rf_old(train_images, train_labels, test_images, test_labels, fraction_of_train_samples, class1, class2):
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


def run_one_layer_deep_conv_rf(train_images, train_labels, test_images, test_labels, fraction_of_train_samples, class1, class2):
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


def run_two_layer_deep_conv_rf_old(train_images, train_labels, test_images, test_labels, fraction_of_train_samples, class1, class2):
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

    # ConvRF (layer 2)
    conv2 = ConvRFOld(kernel_size=7, stride=1)
    conv2_map = conv2.convolve_fit(conv1_map, train_labels)

    # Full RF
    conv1_full_RF = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    conv1_full_RF.fit(conv2_map.reshape(len(train_images), -1), train_labels)

    # Test (after ConvRF 1 and Full RF)
    conv1_map_test = conv1.convolve_predict(test_images)
    conv2_map_test = conv2.convolve_predict(conv1_map_test)
    test_preds = conv1_full_RF.predict(conv2_map_test.reshape(len(test_images), -1))

    return accuracy_score(test_labels, test_preds)


def run_two_layer_deep_conv_rf(train_images, train_labels, test_images, test_labels, fraction_of_train_samples, class1, class2):
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
    conv1_full_RF = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    conv1_full_RF.fit(conv2_map.reshape(len(train_images), -1), train_labels)

    # Test (after ConvRF 1 and Full RF)
    conv1_map_test = conv1.convolve_predict(test_images)
    conv2_map_test = conv2.convolve_predict(conv1_map_test)
    test_preds = conv1_full_RF.predict(conv2_map_test.reshape(len(test_images), -1))

    return accuracy_score(test_labels, test_preds)

###############################################################################
# Wrapper Code for Different CNN Experiments
###############################################################################


BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCH = 100

# transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root=cifar_data_path, train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(
    root=cifar_data_path, train=False, download=True, transform=transform)


def cnn_train_model(model, train_loader, test_loader, optimizer, scheduler, EPOCH, config):
    t0 = time.perf_counter()

    Loss_train = np.zeros((EPOCH,))
    Loss_test = np.zeros((EPOCH,))
    Acc_test = np.zeros((EPOCH,))
    Acc_train = np.zeros((EPOCH,))
    Time_test = np.zeros((EPOCH,))

    for epoch in range(EPOCH):
        scheduler.step()

        # train 1 epoch
        model.train()
        correct = 0
        train_loss = 0
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x)
            b_y = Variable(y)
            scores = model(b_x)
            loss = F.nll_loss(scores, b_y)      # negative log likelyhood
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step()                    # apply gradients
            model.zero_grad()

            # computing training accuracy
            pred = scores.data.max(1, keepdim=True)[1]
            correct += pred.eq(b_y.data.view_as(pred)).long().cpu().sum()
            train_loss += F.nll_loss(scores, b_y, reduction='sum').item()

        Acc_train[epoch] = 100 * float(correct) / float(len(train_loader.dataset))
        Loss_train[epoch] = train_loss / len(train_loader.dataset)

        # testing
        model.eval()
        correct = 0
        test_loss = 0
        for step, (x, y) in enumerate(test_loader):
            b_x = Variable(x)
            b_y = Variable(y)
            scores = model(b_x)
            test_loss += F.nll_loss(scores, b_y, reduction='sum').item()
            pred = scores.data.max(1, keepdim=True)[1]
            correct += pred.eq(b_y.data.view_as(pred)).long().cpu().sum()

        Loss_test[epoch] = test_loss/len(test_loader.dataset)
        Acc_test[epoch] = 100 * float(correct) / float(len(test_loader.dataset))
        Time_test[epoch] = time.perf_counter()-t0

    return Acc_test[-1]

###########################################################################################################
# CNN Model Trainer Helper
###########################################################################################################


def cnn_train_test(model, fraction_of_train_samples, config, class1, class2):
    print("Experiment:", str(config))

    # set params
    learning_rate = config["lr"]
    weight_decay = config["weight_decay"]

    # get only train images and labels for two classes
    cifar_train_labels = trainset.targets
    cifar_test_labels = testset.targets

    indx_0 = np.argwhere(np.asarray(cifar_train_labels) == class1).flatten()
    indx_0 = indx_0[:int(len(indx_0) * fraction_of_train_samples)]
    indx_1 = np.argwhere(np.asarray(cifar_train_labels) == class2).flatten()
    indx_1 = indx_1[:int(len(indx_1) * fraction_of_train_samples)]
    indx = np.concatenate([indx_0, indx_1])

    trainset_sub = copy.deepcopy(trainset)
    trainset_sub.data = trainset_sub.data[indx, :, :, :]
    trainset_sub.targets = np.asarray(trainset_sub.targets)[indx]
    trainset_sub.targets[trainset_sub.targets == class1] = 0
    trainset_sub.targets[trainset_sub.targets == class2] = 1

    indx_0 = np.asarray(cifar_test_labels) == class1
    indx_1 = np.asarray(cifar_test_labels) == class2
    indx = indx_0 + indx_1

    testset_sub = copy.deepcopy(testset)
    testset_sub.data = testset_sub.data[indx, :, :, :]
    testset_sub.targets = np.asarray(testset_sub.targets)[indx]
    testset_sub.targets[testset_sub.targets == class1] = 0
    testset_sub.targets[testset_sub.targets == class2] = 1

    train_loader = Data.DataLoader(dataset=trainset_sub, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = Data.DataLoader(dataset=testset_sub, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=EPOCH/3, gamma=.1)

    return cnn_train_model(model, train_loader, test_loader, optimizer, scheduler, EPOCH, config)


def run_cnn(cnn_model, fraction_of_train_samples, cnn_config, class1, class2):
    return cnn_train_test(cnn_model, fraction_of_train_samples, cnn_config, class1, class2)


###############################################################################
# Experiments
###############################################################################

fraction_of_train_samples_space = np.geomspace(0.01, MAX_TRAIN_FRACTION, num=10)


def print_old_results(file_name):
    global fraction_of_train_samples_space
    accuracy_scores = np.load(file_name)
    for fraction_of_train_samples, (best_accuracy, time_taken) in zip(fraction_of_train_samples_space, accuracy_scores):
        print("Train Fraction:", str(fraction_of_train_samples))
        print("Accuracy:", str(best_accuracy))
        print("Experiment Runtime: " + str(time_taken), "\n")
        print()
    return accuracy_scores


def run_experiment(experiment, experiment_result_file, text, cnn_model=None, cnn_config={}, class1=class_one, class2=class_two):
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
                best_accuracy = np.mean(
                    [experiment(cnn_model, fraction_of_train_samples, cnn_config, class1, class2) for _ in range(repeats + 1)])
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


# RFs
naive_rf_acc_vs_n, naive_rf_acc_vs_n_times = list(zip(*run_experiment(
    run_naive_rf, "naive_rf_acc_vs_n", "Naive RF")))
deep_conv_rf_old_acc_vs_n, deep_conv_rf_old_acc_vs_n_times = list(zip(*run_experiment(
    run_one_layer_deep_conv_rf_old, "deep_conv_rf_old_acc_vs_n", "DeepConvRF (unshared)")))
deep_conv_rf_acc_vs_n, deep_conv_rf_acc_vs_n_times = list(zip(*run_experiment(
    run_one_layer_deep_conv_rf, "deep_conv_rf_acc_vs_n", "DeepConvRF (shared)")))
deep_conv_rf_old_two_layer_acc_vs_n, deep_conv_rf_old_two_layer_acc_vs_n_times = list(zip(*run_experiment(
    run_two_layer_deep_conv_rf_old, "deep_conv_rf_old_two_layer_acc_vs_n", "DeepConvRF (2-layer, unshared)")))
deep_conv_rf_two_layer_acc_vs_n, deep_conv_rf_two_layer_acc_vs_n_times = list(zip(*run_experiment(
    run_two_layer_deep_conv_rf, "deep_conv_rf_two_layer_acc_vs_n", "DeepConvRF (2-layer, shared)")))


# CNNs
cnn_acc_vs_n_config = {'model': 0, 'lr': 0.001, 'weight_decay': 1e-05}
cnn_acc_vs_n, cnn_acc_vs_n_times = list(zip(*run_experiment(run_cnn, "cnn_acc_vs_n",
                                                            "CNN (1-filter)", cnn_model=SimpleCNN1layer(1, NUM_CLASSES), cnn_config=cnn_acc_vs_n_config)))
cnn32_acc_vs_n_config = {'model': 1, 'lr': 0.001, 'weight_decay': 0.1}
cnn32_acc_vs_n, cnn32_acc_vs_n_times = list(zip(*run_experiment(run_cnn, "cnn32_acc_vs_n",
                                                                "CNN (32-filter)", cnn_model=SimpleCNN1layer(32, NUM_CLASSES), cnn_config=cnn32_acc_vs_n_config)))
cnn32_two_layer_acc_vs_n_config = {'model': 2, 'lr': 0.001, 'weight_decay': 0.1}
cnn32_two_layer_acc_vs_n, cnn32_two_layer_acc_vs_n_times = list(zip(*run_experiment(run_cnn, "cnn32_two_layer_acc_vs_n",
                                                                                    "CNN (2-layer, 32-filter)", cnn_model=SimpleCNN2Layers(32, NUM_CLASSES), cnn_config=cnn32_two_layer_acc_vs_n_config)))

# cnn_best_acc_vs_n, cnn_best_acc_vs_n_times = list(zip(*run_experiment(run_cnn, "cnn_best_acc_vs_n",
#                                                                       "CNN (best)", cnn_model=BestCNN)))

###############################################################################
# Plots
###############################################################################

plt.rcParams['figure.figsize'] = 15, 12
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['font.size'] = 25
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.handlelength'] = 3
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['lines.linewidth'] = 3

# plot accuracies
total_train_samples = len(np.argwhere(cifar_train_labels == class_one)) + \
    len(np.argwhere(cifar_train_labels == class_two))
x_lables = list(fraction_of_train_samples_space*total_train_samples)

fig, ax = plt.subplots()
ax.plot(x_lables, naive_rf_acc_vs_n, marker="", color='green',
        linestyle=":", label="NaiveRF")

ax.plot(x_lables, deep_conv_rf_old_acc_vs_n, marker="", color='brown',
        linestyle="--", label="DeepConvRF (1-layer, unshared)")
ax.plot(x_lables, deep_conv_rf_old_two_layer_acc_vs_n, marker="",
        color='brown', label="DeepConvRF (2-layer, unshared)")

ax.plot(x_lables, deep_conv_rf_acc_vs_n, marker="", color='green',
        linestyle="--", label="DeepConvRF (1-layer, shared)")
ax.plot(x_lables, deep_conv_rf_two_layer_acc_vs_n, marker="",
        color='green', label="DeepConvRF (2-layer, shared)")

ax.plot(x_lables, np.array(cnn_acc_vs_n)/100.0, marker="", color='orange',
        linestyle=":", label="CNN (1-filter)")
ax.plot(x_lables, np.array(cnn32_acc_vs_n)/100.0, marker="", color='orange',
        linestyle="--", label="CNN (1-layer, 32-filters)")
ax.plot(x_lables, np.array(cnn32_two_layer_acc_vs_n)/100.0, marker="",
        color='orange', label="CNN (2-layer, 32-filters)")

# ax.plot(x_lables, cnn_best_acc_vs_n, marker="", color='blue', label="CNN (best)")


ax.set_xlabel('# of Train Samples', fontsize=18)
ax.set_xscale('log')
ax.set_xticks(x_lables)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax.set_ylabel('Accuracy', fontsize=18)

ax.set_title("1 (Automobile) vs 9 (Truck) Classification", fontsize=18)
plt.legend()
plt.savefig(base_path+"rf_deepconvrf_cnn_comparisons.png")


# plot execution times
fig, ax = plt.subplots()
ax.plot(x_lables, naive_rf_acc_vs_n_times, marker="", color='green',
        linestyle=":", label="NaiveRF")

ax.plot(x_lables, deep_conv_rf_old_acc_vs_n_times, marker="", color='brown',
        linestyle="--", label="DeepConvRF (1-layer, unshared)")
ax.plot(x_lables, deep_conv_rf_old_two_layer_acc_vs_n_times, marker="",
        color='brown', label="DeepConvRF (2-layer, unshared)")

ax.plot(x_lables, deep_conv_rf_acc_vs_n_times, marker="", color='green',
        linestyle="--", label="DeepConvRF (1-layer, shared)")
ax.plot(x_lables, deep_conv_rf_two_layer_acc_vs_n_times, marker="",
        color='green', label="DeepConvRF (2-layer, shared)")

ax.plot(x_lables, cnn_acc_vs_n_times, marker="", color='orange',
        linestyle=":", label="CNN (1-filter)")
ax.plot(x_lables, cnn32_acc_vs_n_times, marker="", color='orange',
        linestyle="--", label="CNN (1-layer, 32-filters)")
ax.plot(x_lables, cnn32_two_layer_acc_vs_n_times, marker="",
        color='orange', label="CNN (2-layer, 32-filters)")

# ax.plot(x_lables, cnn_best_acc_vs_n_times, marker="", color='blue', label="CNN (best)")


ax.set_xlabel('# of Train Samples', fontsize=18)
ax.set_xscale('log')
ax.set_xticks(x_lables)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax.set_ylabel('Execution Time (seconds)', fontsize=18)

ax.set_title("1 (Automobile) vs 9 (Truck) Classification", fontsize=18)
plt.legend()
plt.savefig(base_path+"rf_deepconvrf_cnn_perf_comparisons.png")
