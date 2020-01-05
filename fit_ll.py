import numpy as np
import keras.preprocessing as Kprep
from keras import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from numpy import uint8, float32
from numpy.lib.npyio import NpzFile

from fit import create_dnn

if __name__ == '__main__':
    data: NpzFile = np.load(file='dataset.npz')
    x: np.ndarray = np.zeros_like(a=data['x'], dtype=float32)
    x[:] = data['x'] / 255
    y: np.ndarray = data['y']
    y = Kprep.utils.to_categorical(y=y, dtype=uint8)

    gene: Kprep.image.ImageDataGenerator = Kprep.image.ImageDataGenerator(featurewise_center=False,
                                                                          featurewise_std_normalization=False,
                                                                          samplewise_center=False,
                                                                          samplewise_std_normalization=False,
                                                                          rotation_range=20,
                                                                          width_shift_range=0.1,
                                                                          height_shift_range=0.1,
                                                                          horizontal_flip=False,
                                                                          fill_mode='nearest',
                                                                          rescale=None,
                                                                          validation_split=0.2)
    gene.fit(x=x)
    model: Model = create_dnn(shape=x.shape, nb_classes=y.shape[1])
    model.compile(optimizer=Adam(lr=0.0003, decay=1e-6),
                  loss=categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary()

    model.fit_generator(generator=gene.flow(x=x, y=y, batch_size=2, subset='training', shuffle=True),
                        steps_per_epoch=x.shape[0] * 10,
                        verbose=1,
                        validation_steps=x.shape[0],
                        validation_data=gene.flow(x=x, y=y, subset='validation', shuffle=True, batch_size=32),
                        epochs=30,
                        workers=12)
    model.save(filepath='model_ll.h5')
