"""Example usage of MnistEstimator and MnistPreprocessor for evaluating."""
from __future__ import division
import numpy as np
import tensorflow as tf

from preprocess.example.corrupt import CorruptingPreprocessor
from preprocess.example.mnist import get_labels
from estimator import MnistEstimator


batch_size = 128


def input_fn():
    """Input function for estimator.fit."""
    preprocessor = CorruptingPreprocessor(
        'test', include_indices=True, drop_prop=0.0)
    index, image, labels = preprocessor.get_preprocessed_batch(
        batch_size, num_epochs=1, shuffle=False,)
    image = tf.expand_dims(image, axis=-1)
    return (index, image),


estimator = MnistEstimator()
predictions = list(estimator.predict(input_fn=input_fn))
index, predictions = zip(*[(p['index'], p['prediction']) for p in predictions])

predictions = np.array(predictions, dtype=np.int32)

labels = get_labels('test')
total = len(predictions)
correct = np.sum(np.equal(predictions, labels[[index]]))
accuracy = correct / total
print('Test accuracy: %.4f' % (100*accuracy))
