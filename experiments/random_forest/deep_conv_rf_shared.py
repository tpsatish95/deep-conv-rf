from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from dataset import get_subset_data
from random_forest.deep_conv_rf import DeepConvRF


def run_one_layer_deep_conv_rf_shared(dataset_name, data, choosen_classes, sub_train_indices):
    (train_images, train_labels), (test_images, test_labels) = get_subset_data(
        dataset_name, data, choosen_classes, sub_train_indices)

    # Train
    # ConvRF (layer 1)
    conv1 = DeepConvRF(type="shared", kernel_size=10, stride=2)
    conv1_map = conv1.convolve_fit(train_images, train_labels)

    # Full RF
    conv1_full_RF = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    conv1_full_RF.fit(conv1_map.reshape(len(train_images), -1), train_labels)

    # Test (after ConvRF 1 and Full RF)
    conv1_map_test = conv1.convolve_predict(test_images)
    test_preds = conv1_full_RF.predict(conv1_map_test.reshape(len(test_images), -1))

    return accuracy_score(test_labels, test_preds)


def run_two_layer_deep_conv_rf_shared(dataset_name, data, choosen_classes, sub_train_indices):
    (train_images, train_labels), (test_images, test_labels) = get_subset_data(
        dataset_name, data, choosen_classes, sub_train_indices)

    # Train
    # ConvRF (layer 1)
    conv1 = DeepConvRF(type="shared", kernel_size=10, stride=2)
    conv1_map = conv1.convolve_fit(train_images, train_labels)

    # ConvRF (layer 2)
    conv2 = DeepConvRF(type="shared", kernel_size=7, stride=1)
    conv2_map = conv2.convolve_fit(conv1_map, train_labels)

    # Full RF
    conv1_full_RF = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    conv1_full_RF.fit(conv2_map.reshape(len(train_images), -1), train_labels)

    # Test (after ConvRF 1 and Full RF)
    conv1_map_test = conv1.convolve_predict(test_images)
    conv2_map_test = conv2.convolve_predict(conv1_map_test)
    test_preds = conv1_full_RF.predict(conv2_map_test.reshape(len(test_images), -1))

    return accuracy_score(test_labels, test_preds)
