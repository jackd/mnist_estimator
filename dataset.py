"""tf.contrib.data.Dataset wrapper to mnist input_data."""
import os
import tensorflow as tf
from tensorflow.contrib.data import Dataset
from tensorflow.examples.tutorials.mnist import input_data


class DatasetKeys(object):
    """Standard names for MNIST datasets."""

    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'


def _get_data(dataset_key):
    data = input_data.read_data_sets(
        os.path.join(
            os.path.realpath(os.path.dirname(__file__)), 'MNIST_data'),
        one_hot=False)
    if dataset_key == DatasetKeys.TRAIN:
        data = data.train
    elif dataset_key == DatasetKeys.VALIDATION:
        data = data.validation
    elif dataset_key == DatasetKeys.TEST:
        data = data.test
    return data


def mnist_dataset(dataset_key=DatasetKeys.TRAIN):
    data = _get_data(dataset_key)
    images = tf.constant(data.images.reshape(-1, 28, 28))
    labels = tf.constant(data.labels, dtype=tf.int32)
    # tensors = {
    #     'images': images,
    #     'labels': labels,
    # }
    tensors = images, labels
    return Dataset.from_tensor_slices(tensors)


def get_test_labels():
    return _get_data(DatasetKeys.TEST).labels
