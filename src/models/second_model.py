# https://arxiv.org/abs/1702.05931

import numpy as np
np.random.seed(1337)
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from utils.data_augmentation import DataAugmentation


def MakeModel(img_height, img_width, num_classes):
    model = Sequential([
        DataAugmentation(img_height, img_width).data_augmentation,
        # layers.Resizing(img_height, img_width),
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(32, 5, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 5, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(1024, 9, padding='same', activation='relu'),
        layers.Conv2D(512, 1, padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dense(num_classes)
    ])
    return model
