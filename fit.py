from typing import Tuple, List

import keras
from keras import Model, Sequential
from keras.applications import vgg16
from keras.datasets import cifar10
from keras.layers import Conv2D, Activation, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import RMSprop, Adam
from keras.utils import Sequence
import tensorflow as tf
import numpy as np


def create_dnn(shape: List[int], nb_classes: int) -> Model:
    tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    print(shape, nb_classes)

    filter: int = 64
    kernel_size: Tuple[int, int] = (3, 3)
    pool_size: Tuple[int, int] = (2, 2)
    model: Sequential = Sequential()
    model.add(Conv2D(filters=filter, kernel_size=kernel_size, padding='same',
                     input_shape=shape[1:], activation='relu'))
    model.add(Conv2D(filters=filter, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(Conv2D(filters=filter, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=filter, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(Conv2D(filters=filter, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(Conv2D(filters=filter, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=filter, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(Conv2D(filters=filter, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(Conv2D(filters=filter, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=filter, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(Conv2D(filters=filter, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(Conv2D(filters=filter, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model


def normalize(x: np.ndarray) -> np.ndarray:
    xx: np.ndarray = np.zeros_like(a=x, dtype=float)
    xx[:] = x / 255
    return xx


if __name__ == '__main__':
    (train_x, train_y), (valid_x, valid_y) = cifar10.load_data()
    train_x = normalize(x=train_x)
    train_y = keras.utils.to_categorical(train_y)
    valid_x = normalize(x=valid_x)
    valid_y = keras.utils.to_categorical(valid_y)

    print(np.max(train_x))

    model: Model = create_dnn(shape=train_x.shape, nb_classes=train_y.shape[1])
    model.compile(optimizer=Adam(lr=0.0003, decay=1e-6),
                  loss=categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary()

    model.fit(x=train_x,
              y=train_y,
              validation_data=[valid_x, valid_y],
              batch_size=256,
              epochs=48)
    model.save('model.h5')
