import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from dataset import get_subset_data


def run_naive_rf(dataset_name, data, choosen_classes, sub_train_indices, rf_type="shared"):
    (train_images, train_labels), (test_images, test_labels) = get_subset_data(
        dataset_name, data, choosen_classes, sub_train_indices)
    time_taken = dict()

    # Train
    train_start = time.time()
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    clf.fit(train_images.reshape(len(train_images), -1), train_labels)
    train_end = time.time()
    time_taken["train"] = train_end - train_start

    # Test
    test_start = time.time()
    test_preds = clf.predict(test_images.reshape(len(test_images), -1))
    test_end = time.time()
    time_taken["test"] = test_end - test_start

    return accuracy_score(test_labels, test_preds), time_taken
