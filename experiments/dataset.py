import copy

import numpy as np
import torch.utils.data as Data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def normalize(x):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    return (x - mean) / std


def get_dataset(data_path, dataset_name="CIFAR10", is_numpy=True):
    if is_numpy:
        transformer = None
    else:
        if dataset_name == "CIFAR10":
            transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    if dataset_name == "MNIST":
        trainset = datasets.MNIST(root=data_path, train=True, download=True, transform=transformer)
        testset = datasets.MNIST(root=data_path, train=False, download=True, transform=transformer)

    elif dataset_name == "FashionMNIST":
        trainset = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transformer)
        testset = datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transformer)

    elif dataset_name == "SVHN":
        trainset = datasets.SVHN(root=data_path, split='train', download=True, transform=transformer)
        trainset.train_data = np.transpose(trainset.data, (0, 2, 3, 1))
        trainset.train_labels = trainset.labels

        testset = datasets.SVHN(root=data_path, split='test', download=True, transform=transformer)
        testset.test_data = np.transpose(testset.data, (0, 2, 3, 1))
        testset.test_labels = testset.labels

    else:
        trainset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transformer)
        testset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transformer)

    if is_numpy:
        train_images = normalize(trainset.train_data)
        test_images = normalize(testset.test_data)

        train_labels = np.array(trainset.train_labels)
        test_labels = np.array(testset.test_labels)

        return (train_images, train_labels), (test_images, test_labels)

    else:
        return trainset, testset


def get_subset_data(dataset_name, data, choosen_classes, fraction_of_train_samples, is_numpy=True, batch_size=None):
    if is_numpy:
        num_samples_per_class = [int(np.sum(data["train_labels"] == class_index)
                                     * fraction_of_train_samples) for class_index in choosen_classes]
        train_images = [data["train_images"][data["train_labels"] == class_index]
                        [:num_samples_per_class[i]] for i, class_index in enumerate(choosen_classes)]
        train_images = np.concatenate(train_images)
        train_labels = np.concatenate([np.repeat(i, num_samples)
                                       for i, num_samples in enumerate(num_samples_per_class)])

        test_images = np.concatenate(
            [data["test_images"][data["test_labels"] == class_index] for class_index in choosen_classes])
        test_labels = np.concatenate([np.repeat(i, np.sum(data["test_labels"] == class_index))
                                      for i, class_index in enumerate(choosen_classes)])

        return (train_images, train_labels), (test_images, test_labels)
    else:
        if dataset_name == "SVHN":
            train_labels = data["trainset"].labels
            test_labels = data["testset"].labels
        else:
            train_labels = data["trainset"].train_labels
            test_labels = data["testset"].test_labels

        # get all train class indices
        train_indices = list()
        for class_index in choosen_classes:
            indx = np.argwhere(np.asarray(train_labels) == class_index).flatten()
            indx = indx[:int(len(indx) * fraction_of_train_samples)]
            train_indices.append(indx)
        train_indices = np.concatenate(train_indices)

        # prepare subset trainset
        trainset_sub = copy.deepcopy(data["trainset"])
        if dataset_name == "SVHN":
            trainset_sub.data = trainset_sub.data[train_indices, :, :, :]
            trainset_sub.labels = np.asarray(trainset_sub.labels)[train_indices]
            for i, class_index in enumerate(choosen_classes):
                trainset_sub.labels[trainset_sub.labels == class_index] = i
        else:
            trainset_sub.train_data = trainset_sub.train_data[train_indices, :, :, :]
            trainset_sub.train_labels = np.asarray(trainset_sub.train_labels)[train_indices]
            for i, class_index in enumerate(choosen_classes):
                trainset_sub.train_labels[trainset_sub.train_labels == class_index] = i

        # get all test class indices
        test_indices = np.asarray(test_labels) == choosen_classes[0]
        for class_index in choosen_classes[1:]:
            test_indices += np.asarray(test_labels) == class_index

        # prepare subset testset
        testset_sub = copy.deepcopy(data["testset"])
        if dataset_name == "SVHN":
            testset_sub.data = testset_sub.data[test_indices, :, :, :]
            testset_sub.labels = np.asarray(testset_sub.labels)[test_indices]
            for i, class_index in enumerate(choosen_classes):
                testset_sub.labels[testset_sub.labels == class_index] = i
        else:
            testset_sub.test_data = testset_sub.test_data[test_indices, :, :, :]
            testset_sub.test_labels = np.asarray(testset_sub.test_labels)[test_indices]
            for i, class_index in enumerate(choosen_classes):
                testset_sub.test_labels[testset_sub.test_labels == class_index] = i

        train_loader = Data.DataLoader(dataset=trainset_sub, batch_size=batch_size, shuffle=True)
        test_loader = Data.DataLoader(dataset=testset_sub, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
