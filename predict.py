"""Example usage of MnistEstimator and MnistPreprocessor for evaluating."""
from __future__ import division
import numpy as np
import tensorflow as tf

from dataset import mnist_dataset, DatasetKeys, get_test_labels

batch_size = 128


def input_fn():
    """Input function for estimator.fit."""
    dataset = mnist_dataset(DatasetKeys.TEST).batch(batch_size)
    images, labels = dataset.make_one_shot_iterator().get_next()
    return tf.expand_dims(images, axis=-1),


def main(estimator):
    predictions = list(estimator.predict(input_fn=input_fn))

    np.save('test_predictions.npy', predictions)

    labels = get_test_labels()
    total = len(predictions)
    correct = np.sum(np.equal(predictions, labels))
    accuracy = correct / total
    print('Test accuracy: %.4f' % (100*accuracy))


if __name__ == '__main__':
    from estimator import EstimatorBuilder
    main(EstimatorBuilder().get_estimator())
