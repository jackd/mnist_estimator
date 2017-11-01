"""Finetunes a MobileNet keras model in an estimator framework."""
import os
import tensorflow as tf
from estimator import EstimatorBuilder


class MobileNetBuilder(EstimatorBuilder):
    """
    EstimatorBuilder with inference network from MobileNet.

    Definitely overkill for MNIST.

    Demonstrates using a pretrained Keras model within the Estimator framework.

    See `export_initial_weights` for converting pretrained weights for use.
    """
    def __init__(
            self, model_input_shape=(128, 128, 3), alpha=0.25,
            depth_multiplier=1, n_classes=10, learning_rate=1e-3,
            weights='imagenet', include_top=False, model_name='mobilenet'):
        self._model_input_shape = model_input_shape
        self._alpha = alpha
        self._depth_multiplier = depth_multiplier
        self._n_classes = n_classes
        self._learning_rate = learning_rate
        self._include_top = include_top
        self._weights = weights
        self._load_weights = False
        self._model = None
        self._model_name = model_name

    def get_logits(self, image, training):
        """
        Get logits for prediction.

        Note:
            * `tf.keras.backend.set_learning_phase` call is required.
            * by default, the model is constructed without weights.
                `export_initial_weights`
        """
        tf.keras.backend.set_learning_phase(training)
        image = tf.image.resize_images(image, self._model_input_shape[:2])
        image = tf.concat([image]*self._model_input_shape[2], axis=-1)
        assert(image.shape.as_list()[1:] == list(self._model_input_shape))
        self._model = tf.keras.applications.MobileNet(
            input_shape=self._model_input_shape,
            include_top=self._include_top,
            alpha=self._alpha, depth_multiplier=self._depth_multiplier,
            input_tensor=image,
            weights=self._weights if self._load_weights else None)
        features = self._model.output
        features = tf.contrib.layers.flatten(features)
        logits = tf.layers.dense(features, self._n_classes)
        return logits

    def export_initial_weights(self, input_fn):
        if self._weights is not None:
            print('Exporting initial weights')
            assert(not self._load_weights)
            assert(self._model is None)
            self._load_weights = True

            features, labels = input_fn()
            self.model_fn(
                features, labels, mode=tf.estimator.ModeKeys.TRAIN,
                config=None)
            saver = tf.train.Saver()
            save_path = os.path.join(self.model_dir, 'model')
            sess = tf.keras.backend.get_session()

            # initialize uninitialized
            init_vars = set(tf.global_variables())
            for v in self._model.variables:
                init_vars.remove(v)
            sess.run(tf.variables_initializer(init_vars))

            saver.save(sess, save_path, global_step=0)
            self._load_weights = False
            self._model = None
            tf.reset_default_graph()
            print('Done.')


def get_finetune_builder():
    return MobileNetBuilder()


def get_fresh_builder():
    return MobileNetBuilder(weights=None, model_name='mobilenet_fresh')
