import os
import tensorflow as tf
from mnist_estimator.estimator import EstimatorBuilder
from mnist_estimator.train import get_input_fn

batch_size = 128
n_steps = 1000
save_every = 100
print_every = 10
model_dir = os.path.join(os.path.dirname(__file__), 'model')
checkpoint_path = os.path.join(model_dir, 'model.ckpt')
latest_checkpoint = tf.train.latest_checkpoint(model_dir)

builder = EstimatorBuilder()
graph = tf.Graph()

print('Building graph...')
with graph.as_default():
    features, labels = get_input_fn(batch_size)()
    spec = builder.model_fn(features, labels, mode='train', config=None)
    loss = spec.loss
    train_op = spec.train_op
    step = tf.train.get_or_create_global_step()
    saver = tf.train.Saver()


print('Starting session...')
with tf.Session(graph=graph) as sess:
    if latest_checkpoint is None:
        print('Initializing variables...')
        sess.run(tf.global_variables_initializer())
    else:
        print('Loading variables...')
        saver.restore(sess, latest_checkpoint)

    print('Training...')
    for i in range(n_steps):
        loss_val, step_val, _ = sess.run([loss, step, train_op])
        if (step_val+1) % print_every == 0:
            print('loss at step %d: %.3f' % (step_val, loss_val))
        if (step_val+1) % save_every == 0:
            print('Saving step %d' % step_val)
            saver.save(sess, checkpoint_path, step_val)
