# Program to take input .svs or .tif and give classification
# according to pretrained model

# Written my Marley Sudbury (1838838)
# for CM3203 One Semester Individual Project

# Instructions for use
# (1) to classify a single .tif or .svs image,
# python classify.py image.tif [-index=1]
# (2) to classify all .tif or .svs images in a folder,
# python classify.py -a images [-index=1]

# Step 0. import libraries

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import numpy as np
import sys
import math

# This is required for the pyvips library
# change this for your install location and vips version, and remember to
# use double backslashes
# vipshome = 'C:\\Users\\Marley\\Downloads\\vips-dev-8.12\\bin'
# vipshome = '/home/c1838838/Downloads/libvips'
vipshome = 'C:\\Users\\c1838838\\Downloads\\vips-dev-8.12\\bin'

# set PATH
os.environ['PATH'] = vipshome + os.pathsep + os.environ['PATH']

import pyvips

def squarify(M,val):
    # Adapted from https://stackoverflow.com/a/45989739
    (a,b,c)=M.shape
    if a>b:
        amount = math.floor((a-b)/2)
        padding=((0,0),(amount,amount),(0,0))
    else:
        amount = math.floor((b-a)/2)
        padding=((amount,amount),(0,0),(0,0))
    return np.pad(M,padding,mode='constant',constant_values=val)

def classify_image(path, index):
    # Takes an image and prints the classification and confidence

    # Adapted from https://libvips.github.io/pyvips/intro.html#numpy-and-pil

    format_to_dtype = {
        'uchar': np.uint8,
        'char': np.int8,
        'ushort': np.uint16,
        'short': np.int16,
        'uint': np.uint32,
        'int': np.int32,
        'float': np.float32,
        'double': np.float64,
        'complex': np.complex64,
        'dpcomplex': np.complex128,
    }

    class_names = ['Negative', 'Positive']

    # img = pyvips.Image.tiffload(os.path.join(root, name), page=5)
    img = pyvips.Image.tiffload(path, page=index)
    img_array = np.ndarray(
        buffer=img.write_to_memory(),
        dtype=format_to_dtype[img.format],
        shape=[img.height, img.width, img.bands]
    )

    img_array = tf.image.resize(img_array, [img_height, img_width])
    img_array = squarify(img_array, 0)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

def classify_directory(path):
    # yourpath = os.path.dirname("D:\\Training Data !\\Cam16\\Training\\Normal\\")
    yourpath = path
    for root, dirs, files in os.walk(yourpath, topdown=False):
        for name in files:
            print(name)
            if name[-3:] in ['tif', 'svs']:
                # img = pyvips.Image.tiffload(os.path.join(root, name), page=5)
                classify_image(os.path.join(root, name), 5)

def main():
    # Step 1. load the model that has been trained from a checkpoint file

    # Adapted from https://www.tensorflow.org/tutorials/keras/save_and_load
    # A lot of the code here is duplicated to allow the model to be
    # recreated. This could be improved by creating a create_model()
    # function in another file.

    batch_size = 32
    global img_height
    global img_width

    img_height = 1000
    img_width = 1000

    AUTOTUNE = tf.data.AUTOTUNE

    normalization_layer = layers.Rescaling(1./255)

    num_classes = 2

    # Create a basic model instance
    global model
    from second_model import MakeModel
    model = MakeModel(img_height, img_width, num_classes)

    model.compile(optimizer='Adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.build((None, img_height, img_width, 3))

    checkpoint_path = "D:/model_2_adam_1000/cp.ckpt"
    model.load_weights(checkpoint_path)

    # Step 2. load the image to check

    # .tif file
    # img = tf.keras.utils.load_img(pathlib.Path('D:\\Training Data !\\Adam compressed\\Negative\\22073.jpg'), target_size = (180, 180))

    # Process arguments
    # default_path = os.path.dirname("D:\\Training Data !\\Cam16\\Training\\Normal\\")

    provided_path = sys.argv[1]
    print(provided_path)
    interpreted_path = os.path.split(provided_path)
    print(interpreted_path)
    dir = None
    file = None

    if os.path.isdir(os.path.join(interpreted_path[0],interpreted_path[1],'\\')):
        # The tail of the provided path is a folder
        # This will only be the case if all files are being
        # classified within the folder
        print("dir")
        dir = os.path.join(interpreted_path[0], interpreted_path[1])
        classify_directory(dir)
    if os.path.isfile(os.path.join(interpreted_path[0],interpreted_path[1])):
        # The tail of the provided path is a file
        # This will only be the case if a specific filed
        # is being classified
        print("file")
        dir = interpreted_path[0]
        file = interpreted_path[1]
        classify_image(os.path.join(dir, file), 5)
    # if os.path.isdir(interpreted_path[0]):
    #     # The head of the provided path is a folder
    #     # This will only be the case if the specified image or
    #     # directory of images are in a directory other than the CWD
    #     # The provided path is a folder
    #     print("dir")
    #     dir = 0
    elif dir == file == None:
        # The provided path is neither a file nor a folder
        print("The specified path is not valid :(")


    if '-a' in sys.argv:
        print('Classifying all images in directory')

if __name__ == "__main__":
    main()
