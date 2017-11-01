"""Example usage of MnistEstimator and MnistPreprocessor for training."""
import tensorflow as tf
from dataset import mnist_dataset, DatasetKeys


def get_corrupt_fn(drop_prob):
    if drop_prob == 0:
        return None
    else:
        def corrupt(images, labels):
            drop = tf.to_float(tf.greater(
                tf.random_uniform(images.shape, dtype=tf.float32), drop_prob))
            return images * drop, labels
    return corrupt


def get_input_fn(batch_size, drop_prob=0.5):
    def input_fn():
        """Input function for estimator.train."""
        dataset = mnist_dataset(DatasetKeys.TRAIN)
        dataset = dataset.map(get_corrupt_fn(drop_prob))
        dataset = dataset.shuffle(10000).repeat().batch(batch_size)
        images, labels = dataset.make_one_shot_iterator().get_next()
        images = tf.expand_dims(images, axis=-1)
        return images, labels
    return input_fn


def train(builder, batch_size=128, drop_prob=0.5, max_steps=10000,
          **train_kwargs):
    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = builder.get_estimator()
    estimator.train(
        input_fn=get_input_fn(batch_size, drop_prob), max_steps=max_steps,
        **train_kwargs)


if __name__ == '__main__':
    from estimator import EstimatorBuilder
    train(EstimatorBuilder())
