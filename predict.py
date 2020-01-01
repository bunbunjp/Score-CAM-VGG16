from time import sleep
from typing import List

import keras
from keras import Model
from keras.datasets import cifar10
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import load_model
from numpy import uint8

from fit import create_dnn, normalize
import cv2
import numpy as np
from keras.models import Model


def grad_cam(input_model, x, layer_name):
    '''
    Args:
       input_model: モデルオブジェクト
       x: 画像(array)
       layer_name: 畳み込み層の名前

    Returns:
       jetcam: 影響の大きい箇所を色付けした画像(array)

    '''

    # 前処理
    X = np.expand_dims(x, axis=0)

    X = X.astype('float32')
    preprocessed_input = X / 255.0

    # 予測クラスの算出

    predictions = model.predict(preprocessed_input)
    class_idx = np.argmax(predictions[0])
    class_output = model.output[:, class_idx]

    #  勾配を取得

    conv_output = model.get_layer(layer_name).output  # layer_nameのレイヤーのアウトプット
    grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
    gradient_function = K.function([model.input], [conv_output, grads])  # model.inputを入力すると、conv_outputとgradsを出力する関数

    output, grads_val = gradient_function([preprocessed_input])
    output, grads_val = output[0], grads_val[0]

    # 重みを平均化して、レイヤーのアウトプットに乗じる
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # 画像化してヒートマップにして合成

    print(x.shape)
    cam = cv2.resize(cam, (x.shape[0], x.shape[1]), cv2.INTER_LINEAR)  # 画像サイズは200で処理したので
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # モノクロ画像に疑似的に色をつける
    jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
    jetcam = (np.float32(jetcam) + x / 2)  # もとの画像に合成

    return jetcam


def score_cam(model, img_array, layer_name, max_N=-1):
    cls = np.argmax(model.predict(img_array))
    act_map_array = Model(inputs=model.input,
                          outputs=model.get_layer(layer_name).output).predict(img_array)

    # extract effective maps
    if max_N != -1:
        act_map_std_list = [np.std(act_map_array[0, :, :, k]) for k in range(act_map_array.shape[3])]
        unsorted_max_indices = np.argpartition(-np.array(act_map_std_list), max_N)[:max_N]
        max_N_indices = unsorted_max_indices[np.argsort(-np.array(act_map_std_list)[unsorted_max_indices])]
        act_map_array = act_map_array[:, :, :, max_N_indices]

    input_shape = model.layers[0].output_shape[1:]  # get input shape

    # 1. upsampled to original input size
    act_map_resized_list = [cv2.resize(act_map_array[0, :, :, k], input_shape[:2],
                                       interpolation=cv2.INTER_LINEAR) for k
                            in range(act_map_array.shape[3])]

    # 2. normalize the raw activation value in each activation map into [0, 1]
    act_map_normalized_list = []
    for act_map_resized in act_map_resized_list:
        if np.max(act_map_resized) - np.min(act_map_resized) != 0:
            act_map_normalized = act_map_resized / (np.max(act_map_resized) - np.min(act_map_resized))
        else:
            act_map_normalized = act_map_resized
        act_map_normalized_list.append(act_map_normalized)

    # 3. project highlighted area in the activation map
    # to original input space by multiplying the normalized activation map
    masked_input_list = []
    for act_map_normalized in act_map_normalized_list:
        masked_input = np.copy(img_array)
        for k in range(3):
            masked_input[0, :, :, k] *= act_map_normalized
        masked_input_list.append(masked_input)
    masked_input_array = np.concatenate(masked_input_list, axis=0)

    # 4. feed masked inputs into CNN model and softmax
    pred_from_masked_input_array = softmax(model.predict(masked_input_array))

    # 5. define weight as the score of target class
    weights = pred_from_masked_input_array[:, cls]

    # 6. get final class discriminative localization map as linear weighted combination of all activation maps
    cam = np.dot(act_map_array[0, :, :, :], weights)
    # cam = np.maximum(0, cam)  # Passing through ReLU
    cam /= np.max(cam)  # scale 0 to 1.0

    return cam


def softmax(x):
    f = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return f


if __name__ == '__main__':
    (train_x, train_y), (valid_x, valid_y) = cifar10.load_data()
    valid_y = keras.utils.to_categorical(valid_y)
    valid_x = normalize(x=valid_x)

    model: Model = create_dnn(shape=valid_x.shape, nb_classes=valid_y.shape[1])
    model.load_weights(filepath='model.h5')
    model.compile(optimizer=Adam(lr=0.0003, decay=1e-6),
                  loss=categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary()

    # eval = model.evaluate(x=valid_x, y=valid_y, batch_size=128)
    # print(eval[1])
    target: List[int] = [10, 20, 30, 40]

    for idx in target:
        grad: np.ndarray = np.zeros_like(a=valid_x[idx], dtype=uint8)
        grad[:] = grad_cam(input_model=model, x=valid_x[idx],
                           layer_name='conv2d_12')
        score = score_cam(model=model, img_array=valid_x[idx:idx + 1],
                          layer_name='conv2d_6')
        fig: Figure = plt.figure()
        # grad[grad > 255] = 255
        print('grad is ', np.max(grad), np.min(grad))
        ax: Axes = fig.add_subplot(1, 3, 1)
        ax.imshow(grad)
        ax.axis('off')

        ax: Axes = fig.add_subplot(1, 3, 2)
        ax.axis('off')
        ax.imshow(valid_x[idx])

        ax: Axes = fig.add_subplot(1, 3, 3)
        ax.imshow(score * 255)
        ax.axis('off')
        fig.show()
        plt.show()
        sleep(1.0)
