###########################################################################################################
# Imports
###########################################################################################################
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import time
import progressbar
import warnings
import copy
from multiprocessing import cpu_count
import torch.multiprocessing as mp

plt.style.use('seaborn')

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

cifar_data_path = "data"

BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCH = 100
TRAIN_FRACTION = 1.0
CLASS1 = 1
CLASS2 = 9

###########################################################################################################
# Data Preparation
###########################################################################################################

print('==> Preparing data..')

# transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root=cifar_data_path, train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(
    root=cifar_data_path, train=False, download=True, transform=transform)

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

###########################################################################################################
# Model Trainer
###########################################################################################################


def train_model(model, train_loader, test_loader, optimizer, scheduler, EPOCH, config):

    t0 = time.perf_counter()

    Loss_train = np.zeros((EPOCH,))
    Loss_test = np.zeros((EPOCH,))
    Acc_test = np.zeros((EPOCH,))
    Acc_train = np.zeros((EPOCH,))
    Time_test = np.zeros((EPOCH,))

    bar = progressbar.ProgressBar(min_value=0, max_value=EPOCH-1)
    for epoch in range(EPOCH):
        bar.update(epoch)
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

    return [Acc_train, Acc_test, Loss_train, Loss_test, config]

###########################################################################################################
# Model Trainer Helper
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

    return train_model(model, train_loader, test_loader, optimizer, scheduler, EPOCH, config)


def plot(acc_train, acc_test, loss_train, loss_test):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(acc_train, label='Acc. Train')
    plt.plot(acc_test, label='Acc. Test')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(loss_train, label='Loss Train')
    plt.plot(loss_test, label='Loss Test')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    plt.clf()


# search space
cnn_model_space = [SimpleCNN1layer(1, NUM_CLASSES), SimpleCNN1layer(
    32, NUM_CLASSES), SimpleCNN2Layers(32, NUM_CLASSES)]  # [0, 1, 2]
lr_space = np.geomspace(1e-6, 1e3, num=10)
weight_decay = np.geomspace(1e-6, 1e3, num=10)

param_dicts = list()
for model_id, m in enumerate(cnn_model_space):
    for l in lr_space:
        for w in weight_decay:
            param_dict = {"model": m, "fraction_of_train_samples": TRAIN_FRACTION, "config": {
                "model": model_id, "lr": l, "weight_decay": w}, "class1": CLASS1, "class2": CLASS2}
            param_dicts.append(param_dict)


def cnn_train_test_unpack(args):
    return cnn_train_test(**args)


with mp.Pool(int(cpu_count()/torch.get_num_threads())) as p:
    experiment_logs = p.map(cnn_train_test_unpack, param_dicts)
np.save("grid_search_best_cnn_logs.npy", np.array(experiment_logs))
