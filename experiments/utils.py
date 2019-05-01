import numpy as np


def get_title_and_results_path(dataset_name, choosen_classes, min_train_samples, max_train_samples):
    if dataset_name == "CIFAR10":
        '''CIFAR10'''
        CIFAR10_MAP = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
                       4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
        TITLE = " vs ".join([CIFAR10_MAP[i].capitalize() +
                             " (" + str(i) + ")" for i in choosen_classes])
        RESULTS_PATH = "results/cifar10/" + "vs".join([str(i) for i in choosen_classes]) + "/" + str(
            min_train_samples) + "_to_" + str(max_train_samples) + "/"
    elif dataset_name == "SVHN":
        '''SVHN'''
        TITLE = " vs ".join([str(i) for i in choosen_classes])
        RESULTS_PATH = "results/svhn/" + "vs".join([str(i) for i in choosen_classes]) + "/" + str(
            min_train_samples) + "_to_" + str(max_train_samples) + "/"
    elif dataset_name == "FashionMNIST":
        '''FashionMNIST'''
        FashionMNIST_MAP = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                            'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        TITLE = " vs ".join([FashionMNIST_MAP[i] + " (" + str(i) + ")" for i in choosen_classes])
        RESULTS_PATH = "results/fashion_mnist/" + "vs".join([str(i) for i in choosen_classes]) + "/" + str(
            min_train_samples) + "_to_" + str(max_train_samples) + "/"

    return TITLE, RESULTS_PATH


def get_number_of_train_samples_space(data, choosen_classes, min_train_samples, max_train_samples):
    class_wise_train_indices = [np.argwhere(data[0][1] == class_index).flatten()
                                for class_index in choosen_classes]

    total_train_samples = sum([len(ci) for ci in class_wise_train_indices])

    MIN_TRAIN_FRACTION = min_train_samples / total_train_samples
    MAX_TRAIN_FRACTION = max_train_samples / total_train_samples
    fraction_of_train_samples_space = np.geomspace(MIN_TRAIN_FRACTION, MAX_TRAIN_FRACTION, num=10)

    train_indices = list()
    for frac in fraction_of_train_samples_space:
        sub_sample = list()
        for i, class_indices in enumerate(class_wise_train_indices):
            if not train_indices:
                num_samples = int(len(class_indices) * frac + 0.5)
                sub_sample.append(np.random.choice(class_indices, num_samples, replace=False))
            else:
                num_samples = int(len(class_indices) * frac + 0.5) - len(train_indices[-1][i])
                sub_sample.append(np.concatenate([train_indices[-1][i], np.random.choice(
                    list(set(class_indices) - set(train_indices[-1][i])), num_samples, replace=False)]).flatten())
        train_indices.append(sub_sample)
    train_indices = [np.concatenate(t).flatten() for t in train_indices]

    number_of_train_samples_space = [len(i) for i in train_indices]

    return number_of_train_samples_space
