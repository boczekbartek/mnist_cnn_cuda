from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam


def get_basic_nn_model(input_shape=(28, 28, 1), num_classes=10):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(input_shape[0]*input_shape[0]))
    model.add(Dropout(0.2))
    model.add(Activation('tanh'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(), metrics=['accuracy'])

    return model
