from time import sleep
from typing import List, Tuple

import keras.preprocessing as Kprep
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.optimizers import Adam
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import uint8, float32
from numpy.lib.npyio import NpzFile

from crawl_data import Configure
from fit import create_dnn
from predict import score_cam


def to_name_per_value(pred: np.ndarray) -> Tuple[str, float32]:
    maxid: int = np.argmax(pred)
    return Configure.NAMES[maxid], pred[maxid]


if __name__ == '__main__':
    data: NpzFile = np.load(file='dataset.npz')
    valid_y = data['y']
    valid_y = Kprep.utils.to_categorical(y=valid_y, dtype=uint8)
    valid_x: np.ndarray = np.zeros_like(a=data['x'], dtype=float32)
    valid_x[:] = data['x'] / 255

    model: Model = create_dnn(shape=valid_x.shape, nb_classes=valid_y.shape[1])
    model.load_weights(filepath='model_ll.h5')
    model.compile(optimizer=Adam(lr=0.0003, decay=1e-6),
                  loss=categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary()

    # eval = model.evaluate(x=valid_x, y=valid_y, batch_size=128)
    # print(eval[1])
    target: List[int] = [i * 20 for i in range(5)]

    score_layer: List[str] = [
        'conv2d_1',
        'conv2d_2',
        'conv2d_3',
        'conv2d_4',
        'conv2d_5',
        'conv2d_6',
        'conv2d_7',
        'conv2d_8',
        'conv2d_9',
        'conv2d_10',
        'conv2d_11',
        'conv2d_12',
    ]

    matplotlib.rcParams['font.family'] = 'IPAexGothic'
    for idx in target:
        grad: np.ndarray = np.zeros_like(a=valid_x[idx], dtype=uint8)
        y: np.ndarray = model.predict(x=valid_x[idx:idx + 1])
        name, per_value = to_name_per_value(pred=y[0])
        answer, _ = to_name_per_value(valid_y[idx])
        fig: Figure = plt.figure()
        fig.suptitle('{0} - {2} : {1:.4f}'.format(name, per_value, answer))
        for li, layer in enumerate(score_layer):
            ax: Axes = fig.add_subplot(3, 6, li + 1)
            score = score_cam(model=model,
                              img_array=valid_x[idx: idx + 1],
                              layer_name=layer)
            ax.imshow(score)
        ax: Axes = fig.add_subplot(3, 6, len(score_layer) + 1)
        ax.imshow(valid_x[idx])
        fig.show()
        plt.show()
        sleep(1.0)
