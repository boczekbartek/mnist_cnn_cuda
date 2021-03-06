from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D


def get_small_cnn_model(input_shape=(28, 28, 1),num_classes=10):
    # Three steps to create a CNN
    # 1. Convolution
    # 2. Activation
    # 3. Pooling
    # Repeat Steps 1,2,3 for adding more hidden layers

    # 4. After that make a fully connected network
    # This fully connected network gives ability to the CNN
    # to classify the samples
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    # Fully connected layer
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes))

    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(), metrics=['accuracy'])
    model.count_params()
    return model
