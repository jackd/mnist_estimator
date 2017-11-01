"""Example usage of MnistEstimator and MnistPreprocessor for evaluating."""
import tensorflow as tf
from dataset import mnist_dataset, DatasetKeys


batch_size = 128


def input_fn():
    """Input function for estimator.fit."""
    dataset = mnist_dataset(DatasetKeys.VALIDATION).batch(batch_size)
    images, labels = dataset.make_one_shot_iterator().get_next()
    images = tf.expand_dims(images, axis=-1)
    return images, labels


if __name__ == '__main__':
    from estimator import EstimatorBuilder
    estimator = EstimatorBuilder().get_estimator()
    evaluation = estimator.evaluate(input_fn=input_fn)
    print(evaluation)
