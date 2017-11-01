"""Provides mnist_estimator function for generating tf.estimator.Estimators."""
import os
import tensorflow as tf


class EstimatorBuilder(object):
    def __init__(self, dense_nodes=(64,), conv_filters=(8, 16), n_classes=10,
                 learning_rate=1e-3, model_name='base'):
        self._dense_nodes = dense_nodes
        self._conv_filters = conv_filters
        self._n_classes = n_classes
        self._learning_rate = learning_rate
        self._model_name = model_name

    def get_logits(self, image, training):
        if len(image.shape) != 4:
            raise ValueError('image tensor must be 4d')
        conv_initializer = tf.contrib.layers.xavier_initializer()
        fc_initializer = tf.contrib.layers.xavier_initializer()
        x = image
        for n in self._conv_filters:
            x = tf.layers.conv2d(
                x, n, 3, padding='SAME', activation=tf.nn.relu,
                kernel_initializer=conv_initializer)
            x = tf.layers.batch_normalization(
                x, scale=False, training=training)
        x = tf.contrib.layers.flatten(x)
        for n in self._dense_nodes:
            x = tf.layers.dense(
                x, n, activation=tf.nn.relu,
                kernel_initializer=fc_initializer)
            x = tf.layers.batch_normalization(
                x, scale=False, training=training)
        x = tf.layers.dense(
            x, self._n_classes, activation=None,
            kernel_initializer=fc_initializer)
        return x

    def get_confidences(self, logits):
        return tf.nn.softmax(logits)

    def get_loss(self, logits, labels):
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits), name='loss')
        return loss

    def get_train_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        step = tf.train.get_or_create_global_step()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, step)
        return train_op

    def get_predictions(self, logits):
        return tf.argmax(logits, axis=-1)

    def get_accuracy(self, labels, predictions):
        accuracy = tf.metrics.accuracy(labels, predictions)
        tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def model_fn(self, features, labels, mode, config):
        images = features
        training = mode == tf.estimator.ModeKeys.TRAIN
        logits = self.get_logits(images, training=training)
        if training:
            loss = self.get_loss(logits, labels)
            train_op = self.get_train_op(loss)
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            loss = self.get_loss(logits, labels)
            predictions = self.get_predictions(logits)
            accuracy = self.get_accuracy(labels, predictions)
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops={'accuracy': accuracy})
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = self.get_predictions(logits)
            return tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions)
        else:
            raise ValueError('mode unrecognized: %s' % mode)

    def get_estimator(self):
        return tf.estimator.Estimator(self.model_fn, self.model_dir)

    @property
    def model_name(self):
        return self._model_name

    @property
    def model_dir(self):
        return os.path.join(
            os.path.dirname(__file__), 'models', self.model_name)
