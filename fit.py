"""Example usage of MnistEstimator and MnistPreprocessor for training."""
import tensorflow as tf
from preprocess.example.corrupt import CorruptingPreprocessor
from estimator import MnistEstimator


batch_size = 128
max_steps = 10000


def input_fn():
    """Input function for estimator.fit."""
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=False)
    preprocessor = CorruptingPreprocessor(
        mnist.train, include_indices=False, drop_prop=0.5)
    image, labels = preprocessor.get_preprocessed_batch(
        batch_size, shuffle=True)
    image = tf.expand_dims(image, axis=-1)
    return image, labels


estimator = MnistEstimator()
estimator.fit(input_fn=input_fn, max_steps=max_steps)
