from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from dataset import get_subset_data


def run_naive_rf(dataset_name, data, choosen_classes, sub_train_indices):
    (train_images, train_labels), (test_images, test_labels) = get_subset_data(
        dataset_name, data, choosen_classes, sub_train_indices)

    # Train
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    clf.fit(train_images.reshape(len(train_images), -1), train_labels)

    # Test
    test_preds = clf.predict(test_images.reshape(len(test_images), -1))

    return accuracy_score(test_labels, test_preds)
