import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


class ConvRF(object):
    def __init__(self, kernel_size=5, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.kernel_forests = None
        self.num_outputs = None

    def _convolve(self, images, labels=None, flatten=False):

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
            _, num_outputs = labels.shape
            out_labels = np.zeros((batch_size, out_dim, out_dim, num_outputs))
            out_labels[:, ] = labels.reshape(batch_size, 1, 1, num_outputs)

        return out_images, out_labels, out_dim

    def convolve_fit(self, images, labels):
        sub_images, sub_labels, out_dim = self._convolve(images, labels=labels, flatten=True)
        _, self.num_outputs = labels.shape

        self.kernel_forests = [[0]*out_dim for _ in range(out_dim)]
        convolved_image = np.zeros((images.shape[0], out_dim, out_dim, self.num_outputs))
        for i in range(out_dim):
            for j in range(out_dim):
                self.kernel_forests[i][j] = RandomForestClassifier()
                self.kernel_forests[i][j].fit(sub_images[:, i, j], sub_labels[:, i, j])
                convolved_image[:, i, j] = self.kernel_forests[i][j].predict(sub_images[:, i, j])

        return convolved_image

    def convolve_predict(self, images):
        if not self.kernel_forests:
            raise Exception("Should fit training data before predicting")

        sub_images, _, out_dim = self._convolve(images, flatten=True)

        kernel_predictions = np.zeros((images.shape[0], out_dim, out_dim, self.num_outputs))
        for i in range(out_dim):
            for j in range(out_dim):
                kernel_predictions[:, i, j] = self.kernel_forests[i][j].predict(sub_images[:, i, j])

        return kernel_predictions
