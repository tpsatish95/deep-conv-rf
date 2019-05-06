import time
from multiprocessing import cpu_count

from sklearn.metrics import accuracy_score

from dataset import get_subset_data
from RerF import fastPredict, fastRerF

TREE_TYPE = "binnedBase"


def run_naive_rerf(dataset_name, data, choosen_classes, sub_train_indices, rf_type="shared"):
    (train_images, train_labels), (test_images, test_labels) = get_subset_data(
        dataset_name, data, choosen_classes, sub_train_indices)
    time_taken = dict()

    # Train
    train_start = time.time()
    forest = fastRerF(X=train_images.reshape(len(train_images), -1),
                      Y=train_labels,
                      forestType=TREE_TYPE,
                      trees=100,
                      numCores=cpu_count() - 1)
    train_end = time.time()
    time_taken["train"] = train_end - train_start
    # forest.printParameters()

    # Test
    test_start = time.time()
    test_preds = fastPredict(test_images.reshape(len(test_images), -1), forest)
    test_end = time.time()
    time_taken["test"] = test_end - test_start

    return accuracy_score(test_labels, test_preds), time_taken
