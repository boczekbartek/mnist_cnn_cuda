import numpy as np
from keras.datasets import mnist

from keras.utils import np_utils

def load_mnist_data() -> (np.array, np.array, np.array, np.array):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    number_of_classes = 10

    Y_train = np_utils.to_categorical(y_train, number_of_classes)
    Y_test = np_utils.to_categorical(y_test, number_of_classes)
    return (X_train, Y_train), (X_test, Y_test)