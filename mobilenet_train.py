import os
from mobilenet_estimator import get_finetune_builder, get_fresh_builder
from train import train, get_input_fn


def train_finetune():
    builder = get_finetune_builder()
    model_dir = builder.model_dir
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    if len(os.listdir(model_dir)) == 0:
        builder.export_initial_weights(get_input_fn(1))
    train(builder)


def train_fresh():
    train(get_fresh_builder())


if __name__ == '__main__':
    train_finetune()
    # train_fresh()
