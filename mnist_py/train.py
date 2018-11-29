#!/usr/bin/env python3.6
import logging
import argparse
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from utils import save_model, save_plots
from data import load_mnist_data
from models import get_big_cnn_model, get_small_cnn_model, get_basic_nn_model

models_initializers = {
    'big_cnn': get_big_cnn_model,
    'small_cnn': get_small_cnn_model,
    'basic_nn': get_basic_nn_model
}


def train(model_name: str, batch_size: int = 64, epochs: int = 5):
    logging.info("Loading MNIST data")
    (X_train, Y_train), (X_test, Y_test) = load_mnist_data()
    num_classes = len(np.unique(Y_train, axis=0))
    logging.info(f"Image shape: {X_train[0].shape}, number of classes: {num_classes}")

    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    logging.info("Creating images generators with augmentation.")
    gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08,
                             shear_range=0.3, height_shift_range=0.08,
                             zoom_range=0.08)

    test_gen = ImageDataGenerator()

    train_generator = gen.flow(X_train, Y_train, batch_size=batch_size)
    test_generator = test_gen.flow(X_test, Y_test, batch_size=batch_size)

    logging.info(f"Initializing model: {model_name}")

    model = models_initializers[model_name](input_shape=X_train[0].shape,
                                            num_classes=num_classes)

    logging.info("Starting model training.")
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_size // batch_size,
                                  epochs=epochs,
                                  validation_data=test_generator,
                                  validation_steps=test_size // batch_size)
    print(history)
    save_plots(history, model_name)
    save_model(model, model_name)


def parse_args():
    parser = argparse.ArgumentParser(description='Train NN models on MNIST database',
                                     formatter_class=argparse.RawTextHelpFormatter)
    models_list = "\n".join([f'* {model}' for model in models_initializers.keys()])
    parser.add_argument('models_list', nargs='*', default=list(models_initializers.keys()),
                        help=f'List of models to train. \nAvailable: \n{models_list}\n'
                        f'Architectures are defined in mnist_py/models')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    for model in args.models_list:
        train(model)
