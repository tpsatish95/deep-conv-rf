from multiprocessing import cpu_count

from sklearn.metrics import accuracy_score

from dataset import get_subset_data
from RerF import fastPredict, fastRerF

TREE_TYPE = "binnedBase"


def run_naive_rerf(dataset_name, data, choosen_classes, sub_train_indices):
    (train_images, train_labels), (test_images, test_labels) = get_subset_data(
        dataset_name, data, choosen_classes, sub_train_indices)

    # Train
    forest = fastRerF(X=train_images.reshape(len(train_images), -1),
                      Y=train_labels,
                      forestType=TREE_TYPE,
                      trees=100,
                      numCores=cpu_count() - 1)
    # forest.printParameters()

    # Test
    test_preds = fastPredict(test_images.reshape(len(test_images), -1), forest)
    return accuracy_score(test_labels, test_preds)
