"""Example usage of MnistEstimator and MnistPreprocessor for evaluating."""
import tensorflow as tf
from preprocess.example.corrupt import CorruptingPreprocessor
from estimator import MnistEstimator


batch_size = 128


def input_fn():
    """Input function for estimator.fit."""
    preprocessor = CorruptingPreprocessor(
        'validation', include_indices=False, drop_prop=0.0)
    image, labels = preprocessor.get_preprocessed_batch(
        batch_size, num_epochs=1, shuffle=False,)
    image = tf.expand_dims(image, axis=-1)
    return image, labels


estimator = MnistEstimator()
evaluation = estimator.evaluate(input_fn=input_fn)
print(evaluation)
