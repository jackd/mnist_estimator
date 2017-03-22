"""Example estimator."""
import os
import tensorflow as tf
import tensorflow.contrib.learn as learn


def _unpack_features(features):
    """
    Unpack index, image from features.

    If features is a tensor, it is interpretted as the image, with index None.
    If features is a dict, image is features['image'] and index is
        features['index'] or None if index is not a key.
    If features is a list/tuple, it must have length 2, and is interpretted as
        [index, image].

    Returns:
        index, image

    Raises:
        TypeError if features is not a tf.Tensor, dict, list or tuple.
    """
    if isinstance(features, tf.Tensor):
        index = None
        image = features
    elif isinstance(features, dict):
        index = features['index'] if index in features else None
        image = features['image']
    elif isinstance(features, (list, tuple)):
        index, image = features
    else:
        raise TypeError('features must be a tf.Tensor or dict')
    return index, image


class MnistEstimator(learn.BaseEstimator):
    """
    Example estimator for MNIST classification.

    Uses two 3x3 conv layers with batch normalization followed by 2 fc layers.

    See `fit.py`, 'eval.py' and `predict.py` for example usage.
    """

    _n_classes = 10
    _learning_rate = 1e-3
    _conv_filters = [8, 16]
    _fc_nodes = [64]

    def __init__(self):
        """Initialize the estimator."""
        super(MnistEstimator, self).__init__(
            os.path.join(os.path.dirname(__file__), 'model'))

    def _get_logits(self, image):
        if len(image.shape) != 4:
            raise ValueError('image tensor must be 4d')
        conv_initializer = tf.contrib.layers.xavier_initializer()
        fc_initializer = tf.contrib.layers.xavier_initializer()
        x = image
        for n in self._conv_filters:
            x = tf.layers.conv2d(
                x, n, 3, padding='SAME', activation=tf.nn.relu,
                kernel_initializer=conv_initializer)
            x = tf.layers.batch_normalization(x, scale=False)
        x = tf.contrib.layers.flatten(x)
        for n in self._fc_nodes:
            x = tf.layers.dense(
                x, n, activation=tf.nn.relu,
                kernel_initializer=fc_initializer)
            x = tf.layers.batch_normalization(x, scale=False)
        x = tf.layers.dense(
            x, self._n_classes, activation=None,
            kernel_initializer=fc_initializer)
        return x

    def _get_confidences(self, logits):
        return tf.nn.softmax(logits)

    def _get_loss(self, logits, labels):
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits), name='loss')
        tf.summary.scalar('loss', loss)
        return loss

    def _get_optimization_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        steps = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)
        if len(steps) == 1:
            step = steps[0]
        else:
            raise Exception('Multiple global steps disallowed')
        train_op = optimizer.minimize(loss, step)
        return train_op

    def _get_predictions(self, logits):
        return tf.argmax(logits, axis=-1)

    def _get_accuracy(self, labels, predictions):
        accuracy = tf.metrics.accuracy(labels, predictions)
        tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def _get_train_ops(self, features, labels):
        index, image = _unpack_features(features)
        tf.summary.image('image', image)
        logits = self._get_logits(image)
        predictions = self._get_predictions(logits)
        loss = self._get_loss(logits, labels)

        train_op = self._get_optimization_op(loss)

        return learn.ModelFnOps(
            learn.ModeKeys.TRAIN,
            predictions=predictions,
            loss=loss,
            train_op=train_op)

    def _get_eval_ops(self, features, labels, metrics):
        index, image = _unpack_features(features)
        logits = self._get_logits(image)
        loss = self._get_loss(logits, labels)
        predictions = self._get_predictions(logits)
        accuracy = self._get_accuracy(labels, predictions)
        eval_metric_ops = {'accuracy': accuracy}
        train_op = []  # unsure why this is necessary for evaluation...
        return learn.ModelFnOps(
            learn.ModeKeys.TRAIN,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops,
            train_op=train_op,
        )

    def _get_predict_ops(self, features):
        index, image = _unpack_features(features)
        logits = self._get_logits(image)
        predictions = self._get_predictions(logits)
        if index is not None:
            predictions = {
                'index': index,
                'prediction': predictions,
            }
        return learn.ModelFnOps(
            learn.ModeKeys.INFER,
            predictions=predictions)
