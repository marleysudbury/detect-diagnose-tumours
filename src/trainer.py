# Train neural network on specified data

# Written my Marley Sudbury (1838838)
# for CM3203 One Semester Individual Project

# Adapted from https://www.tensorflow.org/tutorials/images/classification

# import matplotlib.pyplot as plt
import numpy as np
import sys
import os
# import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
batch_size = 64
img_height = 100
img_width = 100

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(factor=(-1, 1)),
    layers.RandomZoom(height_factor=(-1, 1)),
  ]
)

from second_model import MakeModel

# Import my own modules

# from path_handler import PathHandler
# from image_pipeline import ImagePipeline
#
# path = PathHandler(sys.argv[1])
# pipeline = ImagePipeline()

# data_dir = pathlib.Path('E:\\Training Data !\\Adam compressed')
data_dir = pathlib.Path('/media/c1838838/REM3/Training Data !/Head_Neck_Patch')

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)

model = MakeModel(img_height, img_width, num_classes)

# TODO: try different kernel size, e.g. 10 (orig 3)
# model = Sequential([
#   data_augmentation,
#   layers.Resizing(img_height, img_width),
#   layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Dropout(0.2),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_classes)
# ])

model.compile(optimizer='Adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.build((None, img_height, img_width, 3))

model.summary()

# Save the model so that it can be loaded later
# Adapted from https://www.tensorflow.org/tutorials/keras/save_and_load

checkpoint_path = "/media/c1838838/REM3/model_2_adam_100_patch/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

epochs=200
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[cp_callback]
)
