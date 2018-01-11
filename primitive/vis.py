import os
import tensorflow as tf
from mnist_estimator.estimator import EstimatorBuilder
from mnist_estimator.predict import get_input_fn
import matplotlib.pyplot as plt

batch_size = 128
n_steps = 1000
model_dir = os.path.join(os.path.dirname(__file__), 'model')
checkpoint_path = os.path.join(model_dir, 'model.ckpt')
latest_checkpoint = tf.train.latest_checkpoint(model_dir)
if latest_checkpoint is None:
    raise RuntimeError('No saved model.')

builder = EstimatorBuilder()
graph = tf.Graph()

with graph.as_default():
    features, = get_input_fn(batch_size)()
    spec = builder.model_fn(features, None, mode='infer', config=None)
    predictions = spec.predictions

saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    saver.restore(sess, latest_checkpoint)

    for i in range(n_steps):
        pred_vals, image_vals = sess.run([predictions, features])
        image_vals = image_vals[..., 0]

        for p, im in zip(pred_vals, image_vals):
            plt.imshow(im)
            plt.title(str(p))
            plt.show()
