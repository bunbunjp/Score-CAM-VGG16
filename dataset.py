import glob
import os
import random
from os import mkdir
from sys import path
from time import sleep

import numpy as np
from PIL import Image
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import uint8

from crawl_data import Configure
import cv2
import sys
import os.path


def trim_image(path: str) -> List[np.ndarray]:
    origin: np.ndarray = cv2.imread(path,
                                    cv2.IMREAD_COLOR)
    origin[:] = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
    face_casade: cv2.CascadeClassifier = cv2.CascadeClassifier('lbpcascade_animeface/lbpcascade_animeface.xml')

    gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    face_rect: np.ndarray = face_casade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

    result: List[np.ndarray] = list()
    for (x, y, w, h) in face_rect:
        face: np.ndarray = origin[y: y + h, x: x + w]
        resize: np.ndarray = cv2.resize(face, Configure.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        result.append(resize)
    return result


if __name__ == '__main__':
    IMAGE_DIR: str = 'images'

    train: List[Tuple[np.ndarray, int]] = []

    counter = 0
    MAX_RANGE: int = 50
    random.seed(MAX_RANGE)
    NB_CLASSES: int = len(Configure.NAMES)
    for idx, _ in enumerate(Configure.NAMES):
        files: List[str] = glob.glob(pathname=os.path.join(IMAGE_DIR, str(idx), '*'))
        icnt: int = 0
        for pidx, path in enumerate(files):
            faces: List[np.ndarray] = trim_image(path)
            length: int = len(faces)
            for face in faces:
                train.append((cv2.resize(src=face,
                                         dsize=Configure.IMAGE_SIZE,
                                         interpolation=cv2.INTER_LINEAR), idx))
                # x[counter, :, :, :] = cv2.resize(src=face,
                #                                  dsize=Configure.IMAGE_SIZE,
                #                                  interpolation=cv2.INTER_AREA)
                # y[counter] = idx
                icnt += 1
                counter += 1
                if icnt > MAX_RANGE - 1:
                    break
            if icnt > MAX_RANGE - 1:
                break
        print(counter)
    random.shuffle(x=train)
    x: np.ndarray = np.zeros(shape=(MAX_RANGE * NB_CLASSES, Configure.IMAGE_SIZE[0], Configure.IMAGE_SIZE[1], 3),
                             dtype=uint8)
    y: np.ndarray = np.zeros(shape=(MAX_RANGE * NB_CLASSES),
                             dtype=uint8)
    for idx, tpl in enumerate(train):
        x[idx, :] = tpl[0]
        y[idx] = tpl[1]
    np.savez(file='dataset.npz',
             x=x, y=y)
