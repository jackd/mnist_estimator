"""Example usage of MnistEstimator and MnistPreprocessor for training."""
import tensorflow as tf
from dataset import mnist_dataset, DatasetKeys
from estimator import mnist_estimator

drop_prob = 0.5
batch_size = 128


def get_corrupt_fn(drop_prob):
    if drop_prob == 0:
        return None
    else:
        def corrupt(images, labels):
            drop = tf.to_float(tf.greater(
                tf.random_uniform(images.shape, dtype=tf.float32), drop_prob))
            return images * drop, labels
    return corrupt


def input_fn():
    """Input function for estimator.train."""
    dataset = mnist_dataset(DatasetKeys.TRAIN)
    dataset = dataset.map(get_corrupt_fn(drop_prob))
    dataset = dataset.shuffle(10000).repeat().batch(batch_size)
    images, labels = dataset.make_one_shot_iterator().get_next()
    images = tf.expand_dims(images, axis=-1)
    return images, labels


estimator = mnist_estimator(learning_rate=1e-3)
max_steps = 10000

estimator.train(input_fn=input_fn, max_steps=max_steps)
