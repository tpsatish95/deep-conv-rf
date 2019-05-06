from multiprocessing import cpu_count

from sklearn.metrics import accuracy_score

from dataset import get_subset_data
from random_forest.deep_conv_rf import DeepConvRF
from RerF import fastPredict, fastRerF

NUM_TREES = 1000
TREE_TYPE = "binnedBase"


def run_one_layer_deep_conv_rerf_shared(dataset_name, data, choosen_classes, sub_train_indices):
    (train_images, train_labels), (test_images, test_labels) = get_subset_data(
        dataset_name, data, choosen_classes, sub_train_indices)

    # Train
    # ConvRF (layer 1)
    conv1 = DeepConvRF(type="rerf_shared", kernel_size=10, stride=2, rerf_params={
                       "num_trees": NUM_TREES, "tree_type": TREE_TYPE})
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


def run_two_layer_deep_conv_rerf_shared(dataset_name, data, choosen_classes, sub_train_indices):
    (train_images, train_labels), (test_images, test_labels) = get_subset_data(
        dataset_name, data, choosen_classes, sub_train_indices)

    # Train
    # ConvRF (layer 1)
    conv1 = DeepConvRF(type="rerf_shared", kernel_size=10, stride=2, rerf_params={
                       "num_trees": NUM_TREES, "tree_type": TREE_TYPE})
    conv1_map = conv1.convolve_fit(train_images, train_labels)

    # ConvRF (layer 2)
    conv2 = DeepConvRF(type="rerf_shared", kernel_size=7, stride=1, rerf_params={
                       "num_trees": NUM_TREES, "tree_type": TREE_TYPE})
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
