"""Example usage of MnistEstimator and MnistPreprocessor for training."""
import tensorflow as tf
from preprocess.example.corrupt import CorruptingPreprocessor
from preprocess.example.mnist import MnistDataset
from estimator import mnist_estimator, configure_estimator, is_configured


def input_fn():
    """Input function for estimator.train."""
    # from tensorflow.examples.tutorials.mnist import input_data
    # mnist = input_data.read_data_sets('MNIST_data/', one_hot=False)
    preprocessor = CorruptingPreprocessor(
        MnistDataset.TRAIN, drop_prob=0.5)
    image, labels = preprocessor.get_preprocessed_batch(
        batch_size, shuffle=True)
    image = tf.expand_dims(image, axis=-1)
    return image, labels


model_name = 'base'

if not is_configured(model_name):
    conv_filters = [8, 16]
    dense_nodes = [64]
    configure_estimator(model_name, conv_filters, dense_nodes)
estimator = mnist_estimator(model_name, learning_rate=1e-3)

batch_size = 128
max_steps = 10000

estimator.train(input_fn=input_fn, max_steps=max_steps)
