import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable

from dataset import get_subset_data


def cnn_train_model(model, train_loader, test_loader, optimizer, scheduler, config):
    model = model.to(config["device"])
    if config["device"] == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    t0 = time.perf_counter()

    loss_train = np.zeros((config["epoch"],))
    loss_test = np.zeros((config["epoch"],))
    acc_test = np.zeros((config["epoch"],))
    acc_train = np.zeros((config["epoch"],))
    time_test = np.zeros((config["epoch"],))

    for epoch in range(config["epoch"]):
        scheduler.step()

        # train 1 epoch
        model.train()
        correct = 0
        train_loss = 0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(config["device"]), y.to(config["device"])
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

        acc_train[epoch] = 100 * float(correct) / float(len(train_loader.dataset))
        loss_train[epoch] = train_loss / len(train_loader.dataset)

        # testing
        model.eval()
        correct = 0
        test_loss = 0
        for step, (x, y) in enumerate(test_loader):
            x, y = x.to(config["device"]), y.to(config["device"])
            b_x = Variable(x)
            b_y = Variable(y)
            scores = model(b_x)
            test_loss += F.nll_loss(scores, b_y, reduction='sum').item()
            pred = scores.data.max(1, keepdim=True)[1]
            correct += pred.eq(b_y.data.view_as(pred)).long().cpu().sum()

        loss_test[epoch] = test_loss/len(test_loader.dataset)
        acc_test[epoch] = 100 * float(correct) / float(len(test_loader.dataset))
        time_test[epoch] = time.perf_counter()-t0

    return acc_test[-1]


def run_cnn(dataset_name, model, data, choosen_classes, fraction_of_train_samples, config):
    if "lr" in config:
        learning_rate = config["lr"]
    if "weight_decay" in config:
        weight_decay = config["weight_decay"]

    train_loader, test_loader = get_subset_data(dataset_name, data, choosen_classes, fraction_of_train_samples, is_numpy=False, batch_size=config["batch_size"])

    if "lr" in config and "weight_decay" in config:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=.9, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adadelta(model.parameters())  # used Adadelta, as it wokrs well without any magic numbers

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["epoch"]/3, gamma=.1)

    return cnn_train_model(model, train_loader, test_loader, optimizer, scheduler, config)
