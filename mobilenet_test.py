import numpy as np
import tensorflow as tf

K = tf.keras.backend
K.set_learning_phase(True)


input_shape = (128, 128, 3)
x = tf.ones(shape=(2,) + input_shape, dtype=tf.float32)
model = tf.keras.applications.MobileNet(
    input_shape=input_shape,
    input_tensor=x,
    include_top=False,
    # weights='imagenet',
)
# model(x)
K.get_session()
BASE_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.6/'


def load_weights(input_shape, weights='imagenet', alpha=1, include_top=False):
    rows = input_shape[1]
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            raise ValueError('Weights for "channels_last" format '
                             'are not available.')
        if alpha == 1.0:
            alpha_text = '1_0'
        elif alpha == 0.75:
            alpha_text = '7_5'
        elif alpha == 0.50:
            alpha_text = '5_0'
        else:
            alpha_text = '2_5'

        if include_top:
            model_name = 'mobilenet_%s_%d_tf.h5' % (alpha_text, rows)
            weigh_path = BASE_WEIGHT_PATH + model_name
            weights_path = tf.keras.utils.get_file(
                model_name, weigh_path, cache_subdir='models')
        else:
            model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (
                alpha_text, rows)
            weigh_path = BASE_WEIGHT_PATH + model_name
            weights_path = tf.keras.utils.get_file(
                model_name, weigh_path, cache_subdir='models')
        model.load_weights(weights_path)


print(model.output)

with tf.Session() as sess:
    load_weights(input_shape)
    model_data = sess.run(model.output)


print(np.sum(model_data))
