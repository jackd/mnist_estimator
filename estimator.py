"""Provides mnist_estimator function for generating tf.estimator.Estimators."""
import os
import tensorflow as tf

_n_classes = 10


def _get_logits(image, conv_filters, dense_nodes, n_classes, training):
    if len(image.shape) != 4:
        raise ValueError('image tensor must be 4d')
    conv_initializer = tf.contrib.layers.xavier_initializer()
    fc_initializer = tf.contrib.layers.xavier_initializer()
    x = image
    for n in conv_filters:
        x = tf.layers.conv2d(
            x, n, 3, padding='SAME', activation=tf.nn.relu,
            kernel_initializer=conv_initializer)
        x = tf.layers.batch_normalization(x, scale=False, training=training)
    x = tf.contrib.layers.flatten(x)
    for n in dense_nodes:
        x = tf.layers.dense(
            x, n, activation=tf.nn.relu,
            kernel_initializer=fc_initializer)
        x = tf.layers.batch_normalization(x, scale=False, training=training)
    x = tf.layers.dense(
        x, n_classes, activation=None,
        kernel_initializer=fc_initializer)
    return x


def _get_confidences(self, logits):
    return tf.nn.softmax(logits)


def _get_loss(logits, labels):
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits), name='loss')
    tf.summary.scalar('loss', loss)
    return loss


def _get_train_op(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    steps = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)
    if len(steps) == 1:
        step = steps[0]
    else:
        raise Exception('Multiple global steps disallowed')
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, step)
    return train_op


def _get_predictions(logits):
    return tf.argmax(logits, axis=-1)


def _get_accuracy(labels, predictions):
    accuracy = tf.metrics.accuracy(labels, predictions)
    tf.summary.scalar('accuracy', accuracy)
    return accuracy


def _model_fn(features, labels, mode, params, config):
    images = features
    training = mode == tf.estimator.ModeKeys.TRAIN
    logits = _get_logits(
        images, params['conv_filters'], params['dense_nodes'],
        params['n_classes'], training=training)
    if training:
        loss = _get_loss(logits, labels)
        train_op = _get_train_op(loss, params['learning_rate'])
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        loss = _get_loss(logits, labels)
        predictions = _get_predictions(logits)
        accuracy = _get_accuracy(labels, predictions)
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops={'accuracy': accuracy})
    elif mode == tf.estimator.ModeKeys.PREDICT:
        predictions = _get_predictions(logits)
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        raise Exception('mode unrecognized: %s' % mode)


def mnist_estimator(learning_rate=1e-3):
    """
    Creates a tf.estimator.Estimator.

    Arguments:
        model_name: identifier for this model
        learning_rate: only required for training.

    Returns:
        tf.estimator.Estimator
    """
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    params = dict(
        dense_nodes=[64], conv_filters=[8, 16], n_classes=10,
        learning_rate=learning_rate)
    return tf.estimator.Estimator(_model_fn, model_dir, params=params)
