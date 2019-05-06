import time
from multiprocessing import cpu_count

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from RerF import fastRerF


class DeepConvRF(object):
    def __init__(self, type="unshared", kernel_size=5, stride=2, rerf_params={"num_trees": 1000, "tree_type": "binnedBase"}):
        self.kernel_size = kernel_size
        self.stride = stride
        self.type = type
        self.rerf_params = rerf_params
        self.kernel_forest = None
        self.time_taken = {"load": 0, "train_chop": 0, "test_chop": 0, "train": 0, "fit": 0, "train_post": 0, "test": 0, "test_post": 0}

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

    def convolve_fit(self, images, labels):
        train_chop_start = time.time()
        sub_images, sub_labels = self._convolve_chop(images, labels=labels, flatten=True)
        train_chop_end = time.time()
        self.time_taken["train_chop"] += (train_chop_end - train_chop_start)

        batch_size, out_dim, _, _ = sub_images.shape
        convolved_image = np.zeros((images.shape[0], out_dim, out_dim, 1))

        if self.type == "unshared":
            self.kernel_forest = np.zeros((out_dim, out_dim), dtype=np.int).tolist()

            for i in range(out_dim):
                for j in range(out_dim):
                    fit_start = time.time()
                    self.kernel_forest[i][j] = RandomForestClassifier(n_estimators=32)
                    self.kernel_forest[i][j].fit(sub_images[:, i, j], sub_labels[:, i, j])
                    fit_end = time.time()
                    self.time_taken["fit"] += (fit_end - fit_start)

                    train_post_start = time.time()
                    convolved_image[:, i, j] = self.kernel_forest[i][j].predict_proba(
                        sub_images[:, i, j])[..., 1][..., np.newaxis]
                    train_post_end = time.time()
                    self.time_taken["train_post"] += (train_post_end - train_post_start)

        elif self.type == "shared":
            fit_start = time.time()
            all_sub_images = sub_images.reshape(batch_size*out_dim*out_dim, -1)
            all_sub_labels = sub_labels.reshape(batch_size*out_dim*out_dim, -1)

            self.kernel_forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
            self.kernel_forest.fit(all_sub_images, all_sub_labels)
            fit_end = time.time()
            self.time_taken["fit"] += (fit_end - fit_start)

            train_post_start = time.time()
            for i in range(out_dim):
                for j in range(out_dim):
                    convolved_image[:, i, j] = self.kernel_forest.predict_proba(
                        sub_images[:, i, j])[..., 1][..., np.newaxis]
            train_post_end = time.time()
            self.time_taken["train_post"] += (train_post_end - train_post_start)

        elif self.type == "rerf_shared":
            def approx_predict_proba_sample_wise(sample):
                return np.array(self.kernel_forest.predict_post(sample.tolist())[1] / float(self.rerf_params["num_trees"]))

            fit_start = time.time()
            all_sub_images = sub_images.reshape(batch_size*out_dim*out_dim, -1)
            all_sub_labels = sub_labels.reshape(batch_size*out_dim*out_dim, -1)

            self.kernel_forest = fastRerF(X=all_sub_images,
                                          Y=all_sub_labels,
                                          forestType=self.rerf_params["tree_type"],
                                          trees=self.rerf_params["num_trees"],
                                          numCores=cpu_count() - 1)
            fit_end = time.time()
            self.time_taken["fit"] += (fit_end - fit_start)

            train_post_start = time.time()
            for i in range(out_dim):
                for j in range(out_dim):
                    convolved_image[:, i, j] = np.array([approx_predict_proba_sample_wise(
                        sample) for sample in sub_images[:, i, j]])[..., np.newaxis]
            train_post_end = time.time()
            self.time_taken["train_post"] += (train_post_end - train_post_start)

        self.time_taken["load"] += self.time_taken["train_chop"]
        self.time_taken["train"] += (self.time_taken["fit"] + self.time_taken["train_post"])

        return convolved_image

    def convolve_predict(self, images):
        if not self.kernel_forest:
            raise Exception("Should fit training data before predicting")

        test_chop_start = time.time()
        sub_images, _ = self._convolve_chop(images, flatten=True)
        test_chop_end = time.time()
        self.time_taken["test_chop"] += (test_chop_end - test_chop_start)

        batch_size, out_dim, _, _ = sub_images.shape

        kernel_predictions = np.zeros((images.shape[0], out_dim, out_dim, 1))

        test_post_start = time.time()
        for i in range(out_dim):
            for j in range(out_dim):
                if self.type == "unshared":
                    kernel_predictions[:, i, j] = self.kernel_forest[i][j].predict_proba(
                        sub_images[:, i, j])[..., 1][..., np.newaxis]
                elif self.type == "shared":
                    kernel_predictions[:, i, j] = self.kernel_forest.predict_proba(
                        sub_images[:, i, j])[..., 1][..., np.newaxis]
                elif self.type == "rerf_shared":
                    def approx_predict_proba_sample_wise(sample):
                        return np.array(self.kernel_forest.predict_post(sample.tolist())[1] / float(self.rerf_params["num_trees"]))

                    kernel_predictions[:, i, j] = np.array([approx_predict_proba_sample_wise(
                        sample) for sample in sub_images[:, i, j]])[..., np.newaxis]
        test_post_end = time.time()
        self.time_taken["test_post"] += (test_post_end - test_post_start)

        self.time_taken["load"] += self.time_taken["test_chop"]
        self.time_taken["test"] += self.time_taken["test_post"]

        return kernel_predictions
