"""Provides mnist_estimator function for generating tf.estimator.Estimators."""
import os
import json
import tensorflow as tf

_n_classes = 10


def _model_dir(model_name):
    return os.path.join(os.path.dirname(__file__), 'model', model_name)


def _is_int_list_or_tuple(possible):
    return isinstance(possible, (list, tuple)) and all(
        [isinstance(p, int) for p in possible])


def _verify_build_args(build_args):
    if 'conv_filters' not in build_args:
        raise Exception("build_args contains no 'conv_filters'")
    if not (_is_int_list_or_tuple(build_args['conv_filters'])):
        raise Exception('conv_filters must be list/tuple of ints')
    if 'dense_nodes' not in build_args:
        raise Exception("build_args contains no 'dense_nodes'")
    if not (_is_int_list_or_tuple(build_args['dense_nodes'])):
        raise Exception('dense_nodes must be list/tuple of ints')


def _build_args_path(model_dir):
    return os.path.join(model_dir, 'build_args.json')


def _load_build_args(model_dir):
    path = _build_args_path(model_dir)
    if not os.path.isfile(path):
        raise Exception('No build_args saved at %s' % path)
    with open(path, 'r') as f:
        build_args = json.load(f)
    return build_args


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


def _get_logits(image, conv_filters, dense_nodes, training):
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
        x, _n_classes, activation=None,
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
    indices, images = _unpack_features(features)
    training = mode == tf.estimator.ModeKeys.TRAIN
    logits = _get_logits(
        images, params['conv_filters'], params['dense_nodes'],
        training=training)
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
        if indices is not None:
            predictions = {
                'predictions': predictions,
                'indices': indices
            }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        raise Exception('mode unrecognized: %s' % mode)


def is_configured(model_name='base'):
    """Indicate whether specified model is configured."""
    return os.path.isfile(_build_args_path(_model_dir(model_name)))


def configure_estimator(
        model_name='base', conv_filters=[8, 16], dense_nodes=[64]):
    """
    Configure the estimator.

    Arguments:
        model_name: identifier for this model
        build_args: dict. Required if estimator has yet to be instantiated by
            this function.
        conv_filters: list of ints specifying number of filters used for
            each convolution layer, e.g. [8, 16] results in 2 conv layers
        dense_nodes: list of ints specifying number of nodes in dense
            layers after conv layers, e.g. [64] results in a dense layer from
            input size to 64, then another from 64 to 10 (number of classes).
    """
    build_args = {
        'conv_filters': conv_filters,
        'dense_nodes': dense_nodes
    }
    model_dir = _model_dir(model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    _verify_build_args(build_args)
    path = _build_args_path(model_dir)
    if (os.path.isdir(path)):
        saved_args = _load_build_args(model_dir)
        if saved_args != build_args:
            raise Exception('build_args already exists and is different.')
    else:
        with open(path, 'w') as f:
            json.dump(build_args, f)


def mnist_estimator(model_name='base', learning_rate=1e-3):
    """
    Creates a tf.estimator.Estimator.

    `configure_estimator` must have been called prior with the same model_name.

    Arguments:
        model_name: identifier for this model
        learning_rate: only required for training.

    Returns:
        tf.estimator.Estimator
    """
    if not is_configured(model_name):
        raise Exception(
            'Estimator not configured - run `configure_estimator` first.')
    model_dir = _model_dir(model_name)
    params = _load_build_args(model_dir)
    params['learning_rate'] = learning_rate
    return tf.estimator.Estimator(_model_fn, model_dir, params=params)
