from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.lib.npyio import NpzFile
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data: NpzFile = np.load(file='dataset.npz')
    x: np.ndarray = data['x']
    y: np.ndarray = data['y']

    fig: Figure = plt.figure()
    cnt: int = 0
    for idx in [50, 600, 100, 1000, 0]:
        cnt += 1
        ax: Axes = fig.add_subplot(1, 5, cnt)
        ax.imshow(x[idx])
        ax.set_title(str(y[idx]))
    fig.show()
    plt.show()
    plt.close()
