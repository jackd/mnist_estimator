# MNIST Estimator
This collection of scripts showcases the use of a custom `preprocess.Preprocessor` with a custom `tf.contrib.learn.BaseEstimator`.

## Requirements
* [Tensorflow](https://www.tensorflow.org/) 1.0
* [Numpy](http://www.numpy.org/)
* [preprocess](https://github.com/jackd/preprocess)

## Structure
The model is defined in `estimator.py`. This is provided via a `tf.contrib.learn.BaseEstimator` implementation. The model is very simple, with two `3x3` convolution layers and two fully connected layers, with batch normalization after each convolution and the the first fully connected layer.

### Scripts
* `fit.py`: Trains the data based on the `CorruptingPreprocessor` provided in `preprocess.example.corrupt` for 10000 steps.
* `evaluate.py`: Evaluates the accuracy of the trained estimator on the validation set without any corruption.
* `predict.py`: Calculates the predicted labels for the test images. Also gives compares with the labels to give accuracy (I had to do something with the inferred labels...). The result is similar to `evaluate.py`, though the accuracy is calculated outside of tensorflow.

The model achieves ~98% accuracy after 10000 steps, taking ~10min on a modern gpu with cuda/cudnn support.
