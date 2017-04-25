"""Example usage of MnistEstimator and MnistPreprocessor for evaluating."""
from __future__ import division
import numpy as np
import tensorflow as tf

from preprocess.indexed import IndexedPreprocessor
from preprocess.example.corrupt import CorruptingPreprocessor
from preprocess.example.mnist import get_labels
from estimator import mnist_estimator

batch_size = 128


def input_fn():
    """Input function for estimator.fit."""
    preprocessor = CorruptingPreprocessor(
        'test', drop_prob=0.0)
    preprocessor = IndexedPreprocessor(preprocessor)
    index, image, labels = preprocessor.get_preprocessed_batch(
        batch_size, num_epochs=1, shuffle=False,
        allow_smaller_final_batch=True)
    image = tf.expand_dims(image, axis=-1)
    return (index, image),


estimator = mnist_estimator('base')
predictions = list(estimator.predict(input_fn=input_fn))
indices, predictions = zip(
    *[(p['indices'], p['predictions']) for p in predictions])

predictions = np.array(predictions, dtype=np.int32)

np.save('test_predictions.npy', predictions)

labels = get_labels('test')
total = len(predictions)
correct = np.sum(np.equal(predictions, labels[[indices]]))
accuracy = correct / total
print('Test accuracy: %.4f' % (100*accuracy))
