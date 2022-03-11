from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class DataAugmentation:
    def __init__(self, img_height, img_width):
        self.data_augmentation = Sequential(
          [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                          img_width,
                                          3)),
            layers.RandomRotation(factor=(-1, 1)),
            layers.RandomZoom(height_factor=(-1, 1)),
          ]
        )
