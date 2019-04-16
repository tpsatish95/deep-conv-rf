from multiprocessing import cpu_count

import numpy as np
from sklearn.metrics import accuracy_score

from dataset import get_subset_data
from RerF import fastPredict, fastRerF

NUM_TREES = 1000
TREE_TYPE = "binnedBase"


class DeepConvRerFShared(object):
    def __init__(self, kernel_size=5, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.kernel_forest = None
        self.num_trees = NUM_TREES

    def _convolve_chop(self, images, labels=None, flatten=False):

        batch_size, in_dim, _, num_channels = images.shape

        out_dim = int((in_dim - self.kernel_size) / self.stride) + 1  # calculate output dimensions

        # create matrix to hold the chopped images
        out_images = np.zeros((batch_size, out_dim, out_dim,
                               self.kernel_size, self.kernel_size, num_channels))
        out_labels = None

        curr_y = out_y = 0
        # move kernel vertically across the image
        while curr_y + self.kernel_size <= in_dim:
            curr_x = out_x = 0
            # move kernel horizontally across the image
            while curr_x + self.kernel_size <= in_dim:
                # chop images
                out_images[:, out_x, out_y] = images[:, curr_x:curr_x +
                                                     self.kernel_size, curr_y:curr_y+self.kernel_size, :]
                curr_x += self.stride
                out_x += 1
            curr_y += self.stride
            out_y += 1

        if flatten:
            out_images = out_images.reshape(batch_size, out_dim, out_dim, -1)

        if labels is not None:
            out_labels = np.zeros((batch_size, out_dim, out_dim))
            out_labels[:, ] = labels.reshape(-1, 1, 1)

        return out_images, out_labels

    def rf_predict(self, sample):
        return np.array(self.kernel_forest.predict_post(sample.tolist())[1] / float(self.num_trees))

    def convolve_fit(self, images, labels):
        sub_images, sub_labels = self._convolve_chop(images, labels=labels, flatten=True)

        batch_size, out_dim, _, _ = sub_images.shape

        all_sub_images = sub_images.reshape(batch_size*out_dim*out_dim, -1)
        all_sub_labels = sub_labels.reshape(batch_size*out_dim*out_dim, -1)

        self.kernel_forest = fastRerF(X=all_sub_images,
                                      Y=all_sub_labels,
                                      forestType=TREE_TYPE,
                                      trees=self.num_trees,
                                      numCores=cpu_count() - 1)

        convolved_image = np.zeros((images.shape[0], out_dim, out_dim, 1))
        for i in range(out_dim):
            for j in range(out_dim):
                convolved_image[:, i, j] = np.array([self.rf_predict(sample) for sample in sub_images[:, i, j]])[..., np.newaxis]
        return convolved_image

    def convolve_predict(self, images):
        if not self.kernel_forest:
            raise Exception("Should fit training data before predicting")

        sub_images, _ = self._convolve_chop(images, flatten=True)

        batch_size, out_dim, _, _ = sub_images.shape

        kernel_predictions = np.zeros((images.shape[0], out_dim, out_dim, 1))

        for i in range(out_dim):
            for j in range(out_dim):
                kernel_predictions[:, i, j] = np.array([self.rf_predict(sample) for sample in sub_images[:, i, j]])[..., np.newaxis]
        return kernel_predictions


def run_one_layer_deep_conv_rerf_shared(data, choosen_classes, fraction_of_train_samples):
    (train_images, train_labels), (test_images, test_labels) = get_subset_data(
        data, choosen_classes, fraction_of_train_samples)

    # Train
    # ConvRF (layer 1)
    conv1 = DeepConvRerFShared(kernel_size=10, stride=2)
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


def run_two_layer_deep_conv_rerf_shared(data, choosen_classes, fraction_of_train_samples):
    (train_images, train_labels), (test_images, test_labels) = get_subset_data(
        data, choosen_classes, fraction_of_train_samples)

    # Train
    # ConvRF (layer 1)
    conv1 = DeepConvRerFShared(kernel_size=10, stride=2)
    conv1_map = conv1.convolve_fit(train_images, train_labels)

    # ConvRF (layer 2)
    conv2 = DeepConvRerFShared(kernel_size=7, stride=1)
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
