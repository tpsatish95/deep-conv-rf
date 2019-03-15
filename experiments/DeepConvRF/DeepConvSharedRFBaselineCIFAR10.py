# general imports
import sys
import os.path
import warnings

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torch.optim as optim
import torchvision.transforms as transforms

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sns.set()
warnings.filterwarnings("ignore")
# sys.stdout = open("deep_conv_rf_logs.txt", "w+")

base_path = ""
# base_path = "10_percent_data/"


###########################################################################################################
# Data Preparation
###########################################################################################################

def normalize(x):
    scale = np.mean(np.arange(0, 256))
    return (x - scale) / scale


# train data
cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
cifar_train_images = normalize(cifar_trainset.data)
cifar_train_labels = np.array(cifar_trainset.targets)

# test data
cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
cifar_test_images = normalize(cifar_testset.data)
cifar_test_labels = np.array(cifar_testset.targets)

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


class SimpleCNNOneFilter(torch.nn.Module):

    def __init__(self):
        super(SimpleCNNOneFilter, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, kernel_size=10, stride=2)
        self.fc1 = torch.nn.Linear(144, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 144)
        x = self.fc1(x)
        return(x)


class SimpleCNN32Filter(torch.nn.Module):

    def __init__(self):
        super(SimpleCNN32Filter, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=10, stride=2)
        self.fc1 = torch.nn.Linear(144*32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 144*32)
        x = self.fc1(x)
        return(x)


class SimpleCNN32Filter2Layers(torch.nn.Module):

    def __init__(self):
        super(SimpleCNN32Filter2Layers, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=10, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=7, stride=1)
        self.fc1 = torch.nn.Linear(36*32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 36*32)
        x = self.fc1(x)
        return(x)


class BestCNN(torch.nn.Module):
    def __init__(self):
        super(BestCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

###############################################################################
# Wrapper Code for Different Random Forest Experiments
###############################################################################


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
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    clf.fit(train_images.reshape(-1, 32*32*3), train_labels)
    # Test
    test_preds = clf.predict(test_images.reshape(-1, 32*32*3))
    return accuracy_score(test_labels, test_preds)


def run_one_layer_deep_conv_rf_old(train_images, train_labels, test_images, test_labels, fraction_of_train_samples, class1=3, class2=5):
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
    conv1_full_RF = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    conv1_full_RF.fit(conv1_map.reshape(len(train_images), -1), train_labels)

    # Test (after ConvRF 1 and Full RF)
    conv1_map_test = conv1.convolve_predict(test_images)
    mnist_test_preds = conv1_full_RF.predict(conv1_map_test.reshape(len(test_images), -1))

    return accuracy_score(test_labels, mnist_test_preds)


def run_two_layer_deep_conv_rf_old(train_images, train_labels, test_images, test_labels, fraction_of_train_samples, class1=3, class2=5):
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
    conv1_full_RF = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    conv1_full_RF.fit(conv2_map.reshape(len(train_images), -1), train_labels)

    # Test (after ConvRF 1 and Full RF)
    conv1_map_test = conv1.convolve_predict(test_images)
    conv2_map_test = conv2.convolve_predict(conv1_map_test)
    test_preds = conv1_full_RF.predict(conv2_map_test.reshape(len(test_images), -1))

    return accuracy_score(test_labels, test_preds)

###############################################################################
# Wrapper Code for Different Random Forest Experiments
###############################################################################


# transform
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = datasets.CIFAR10(root='./data', train=True,
                            download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False,
                           download=True, transform=transform)


def cnn_train_test(cnn_model, y_train, y_test, fraction_of_train_samples, class1=3, class2=5):
    # set params
    num_epochs = 5
    learning_rate = 0.001

    class1_indices = np.argwhere(y_train == class1).flatten()
    class1_indices = class1_indices[:int(len(class1_indices) * fraction_of_train_samples)]
    class2_indices = np.argwhere(y_train == class2).flatten()
    class2_indices = class2_indices[:int(len(class2_indices) * fraction_of_train_samples)]
    train_indices = np.concatenate([class1_indices, class2_indices])

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=4, num_workers=2, sampler=train_sampler)

    test_indices = np.concatenate(
        [np.argwhere(y_test == class1).flatten(), np.argwhere(y_test == class2).flatten()])
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=4,
                                              shuffle=False, num_workers=2, sampler=test_sampler)

    # define model
    net = cnn_model()

    # loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.view(-1)).sum().item()
    accuracy = float(correct) / float(total)
    return accuracy


def run_cnn(cnn_model, train_images, train_labels, test_images, test_labels, fraction_of_train_samples, class1=3, class2=5):
    return cnn_train_test(cnn_model, train_labels, test_labels, fraction_of_train_samples, class1, class2)


###############################################################################
# Experiments
###############################################################################

fraction_of_train_samples_space = np.geomspace(0.01, 0.35, num=10)


def print_old_results(file_name):
    global fraction_of_train_samples_space
    accuracy_scores = np.load(file_name)
    for fraction_of_train_samples, best_accuracy in zip(fraction_of_train_samples_space, accuracy_scores):
        print("Train Fraction:", str(fraction_of_train_samples))
        print("Accuracy:", str(best_accuracy))
        print()
    return accuracy_scores


def run_experiment(experiment, experiment_result_file, text, cnn_model=None, class1=3, class2=5):
    global fraction_of_train_samples_space
    print("##################################################################")
    print("acc vs n_samples: " + text + "\n")
    acc_vs_n = list()
    file_name = base_path+experiment_result_file+".npy"
    if not os.path.exists(file_name):
        for fraction_of_train_samples in fraction_of_train_samples_space:
            if not cnn_model:
                best_accuracy = np.mean([experiment(cifar_train_images, cifar_train_labels, cifar_test_images,
                                                    cifar_test_labels, fraction_of_train_samples, class1, class2) for _ in range(2)])
            else:
                best_accuracy = np.mean([experiment(cnn_model, cifar_train_images, cifar_train_labels, cifar_test_images,
                                                    cifar_test_labels, fraction_of_train_samples, class1, class2) for _ in range(2)])
            acc_vs_n.append(best_accuracy)
            print("Train Fraction:", str(fraction_of_train_samples))
            print("Accuracy:", str(best_accuracy), "\n")
        np.save(file_name, acc_vs_n)
    else:
        acc_vs_n = print_old_results(file_name)
    print("##################################################################")

    return acc_vs_n


# RFs
naive_rf_acc_vs_n = run_experiment(
    run_naive_rf, "naive_rf_acc_vs_n", "Naive RF")
deep_conv_rf_old_acc_vs_n = run_experiment(
    run_one_layer_deep_conv_rf_old, "deep_conv_rf_old_acc_vs_n", "DeepConvRF (unshared)")
deep_conv_rf_acc_vs_n = run_experiment(
    run_one_layer_deep_conv_rf, "deep_conv_rf_acc_vs_n", "DeepConvRF (shared)")
deep_conv_rf_old_two_layer_acc_vs_n = run_experiment(
    run_two_layer_deep_conv_rf_old, "deep_conv_rf_old_two_layer_acc_vs_n", "DeepConvRF (2-layer, unshared)")
deep_conv_rf_two_layer_acc_vs_n = run_experiment(
    run_two_layer_deep_conv_rf, "deep_conv_rf_two_layer_acc_vs_n", "DeepConvRF (2-layer, shared)")


# CNNs
cnn_acc_vs_n = run_experiment(run_cnn, "cnn_acc_vs_n",
                              "CNN (1-filter)", cnn_model=SimpleCNNOneFilter)
cnn32_acc_vs_n = run_experiment(run_cnn, "cnn32_acc_vs_n",
                                "CNN (32-filter)", cnn_model=SimpleCNN32Filter)
cnn32_two_layer_acc_vs_n = run_experiment(run_cnn, "cnn32_two_layer_acc_vs_n",
                                          "CNN (2-layer, 32-filter)", cnn_model=SimpleCNN32Filter2Layers)
cnn_best_acc_vs_n = run_experiment(run_cnn, "cnn_best_acc_vs_n",
                                   "CNN (best)", cnn_model=BestCNN)

###############################################################################
# Plots
###############################################################################

plt.rcParams['figure.figsize'] = 15, 12
plt.rcParams['font.size'] = 25
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.handlelength'] = 3
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

fig, ax = plt.subplots()  # create a new figure with a default 111 subplot
ax.plot(fraction_of_train_samples_space*100, naive_rf_acc_vs_n, marker='X', markerfacecolor='red',
        markersize=10, color='green', linewidth=3, linestyle=":", label="NaiveRF")
ax.plot(fraction_of_train_samples_space*100, deep_conv_rf_acc_vs_n, marker='X', markerfacecolor='red',
        markersize=10, color='green', linewidth=3, linestyle="--", label="DeepConvRF (1-layer, shared)")
ax.plot(fraction_of_train_samples_space*100, deep_conv_rf_two_layer_acc_vs_n, marker='X',
        markerfacecolor='red', markersize=10, color='green', linewidth=3, label="DeepConvRF (2-layer, shared)")

ax.plot(fraction_of_train_samples_space*100, deep_conv_rf_old_acc_vs_n, marker='X',
        markerfacecolor='red', markersize=10, color='brown', linewidth=3, linestyle="--", label="DeepConvRF (1-layer, unshared)")
ax.plot(fraction_of_train_samples_space*100, deep_conv_rf_old_two_layer_acc_vs_n, marker='X',
        markerfacecolor='red', markersize=10, color='brown', linewidth=3, label="DeepConvRF (2-layer, unshared)")

ax.plot(fraction_of_train_samples_space*100, cnn_acc_vs_n, marker='X', markerfacecolor='red',
        markersize=10, color='orange', linewidth=3, linestyle=":", label="CNN (1-filter)")
ax.plot(fraction_of_train_samples_space*100, cnn32_acc_vs_n, marker='X', markerfacecolor='red',
        markersize=10, color='orange', linewidth=3, linestyle="--", label="CNN (1-layer, 32-filters)")
ax.plot(fraction_of_train_samples_space*100, cnn32_two_layer_acc_vs_n, marker='X',
        markerfacecolor='red', markersize=10, color='orange', linewidth=3, label="CNN (2-layer, 32-filters)")

ax.plot(fraction_of_train_samples_space*100, cnn_best_acc_vs_n, marker='X',
        markerfacecolor='red', markersize=10, color='blue', linewidth=3, label="CNN (best)")

ax.set_xlabel('Percentage of Train Samples', fontsize=18)
ax.set_xscale('log')
ax.set_xticks(list(fraction_of_train_samples_space*100))
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax.set_ylabel('Accuracy', fontsize=18)
# ax.set_ylim(0.68, 1)

ax.set_title("3 (Cats) vs 5 (Dogs) Classification", fontsize=18)
plt.legend()
plt.savefig(base_path+"rf_deepconvrf_cnn_comparisons.png")
