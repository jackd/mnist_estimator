# MNIST Estimator
This collection of scripts showcases the use of a custom `preprocess.Preprocessor` with a `tf.estimator.Estimator`.

## Requirements
* [Tensorflow](https://www.tensorflow.org/) 1.0
* [Numpy](http://www.numpy.org/)
* [preprocess](https://github.com/jackd/preprocess)

## Structure
The model is defined in `estimator.py`. The default model is very simple, with two `3x3` convolution layers and two fully connected layers, with batch normalization after each convolution and the the first fully connected layer.

### Scripts
* `train.py`: Trains the data based on the `CorruptingPreprocessor` provided in `preprocess.example.corrupt` for 10000 steps.
* `evaluate.py`: Evaluates the accuracy of the trained estimator on the validation set without any corruption.
* `predict.py`: Calculates the predicted labels for the test images and saves the results to file. Also compares with test labels outside of the tensorflow graph/session.

The model achieves ~98% accuracy after 10000 steps, taking ~10min on a modern gpu with cuda/cudnn support.
