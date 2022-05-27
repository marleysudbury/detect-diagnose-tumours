from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from utils.data_augmentation import DataAugmentation


def MakeModel(img_height, img_width, num_classes):
    model = Sequential([
        DataAugmentation(img_height, img_width).data_augmentation,
        layers.Resizing(img_height, img_width),
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    return model
