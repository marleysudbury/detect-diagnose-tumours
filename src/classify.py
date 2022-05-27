# Program to take input .svs or .tif and give classification
# according to pretrained model

# Written by Marley Sudbury (1838838)
# for CM3203 One Semester Individual Project

# Instructions for use
# (1) to classify a single .tif or .svs image,
# python classify.py image.tif [-index=1]
# (2) to classify all .tif or .svs images in a folder,
# python classify.py -a images [-index=1]

# Step 0. import libraries

import os
from utils.load_config import config
openslidehome = config['openslide_path']

os.add_dll_directory(openslidehome)
import openslide
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np
import pathlib
import sys
from utils.normalise_staining import normalizeStaining
from utils.path_handler import PathHandler
from PIL import Image
from utils.image_pipeline import ImagePipeline

# For Grad-CAM visualisation
# https://keras.io/examples/vision/grad_cam/

image_index = 2
stain_normalisation = False

# Data structure to store data for evaluation
classification_confidences = []
actual_classification = "Positive"
tp = 0
fp = 0
tn = 0
fn = 0


def get_img_array(img_path, size):
    # https://keras.io/examples/vision/grad_cam/
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def classify_array(array):
    # Used to classify patch
    # Takes an image and prints the classification and confidence
    # Red is negative
    # Green is positive
    try:
        class_names = ['Negative', 'Positive']
        if stain_normalisation:
            array = np.transpose(np.array(array), axes=[1, 0, 2])
            array = normalizeStaining(img=array)[0]
        img_array = tf.expand_dims(array, 0)  # Create a batch
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        return (int(score[0].numpy() * 100), int(score[1].numpy() * 100), 0)
    except Exception as err:
        print("An error occured while normalising the region")
        print("{}: {}".format(type(err).__name__, err))
        return (0, 0, 0)


def classify_image(pipeline, index):
    # Takes an image and prints the classification and confidence
    try:
        img_array = pipeline.convert_image(index, img_height, img_width, True)
        # if stain_normalisation:
        #     img_array = pipeline.normalise_image(img_array)[0]
        img_array.show()
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        class_names = ['Negative', 'Positive']
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
        return class_names[np.argmax(score)], 100 * np.max(score)
    except Exception as err:
        print("An error occured while classifying the image")
        print("{}: {}".format(type(err).__name__, err))


def classify_directory(pipeline, path):
    for file in path.iterate_files():
        pipeline.new_path(file)
        try:
            label, confidence = classify_image(pipeline, image_index)
        except:
            print("Couldn't load image")

        if label == "Negative":
            confidence = 100 - confidence
        classification_confidences.append(confidence / 100)

    step = 0.01
    for i in np.arange(0.0, 1 + step, step):
        tp = fp = tn = fn = 0
        for con in classification_confidences:
            if con <= i:
                if actual_classification == "Negative":
                    tn += 1
                else:
                    fn += 1
            else:
                if actual_classification == "Negative":
                    fp += 1
                else:
                    tp += 1
        print("At {} threshold. TP: {}, FP: {}, TN: {}, FN: {}".format(
            i, tp, fp, tn, fn))

    print(classification_confidences)

    # AUC evaluation adapted from:
    # https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC

    m = tf.keras.metrics.AUC(num_thresholds=100)
    # correct = [1 for i in range(0, 100)]
    correct = [0 for i in range(0, 100)]
    m.update_state(correct, classification_confidences)
    print(m.result().numpy())


def main():
    if len(sys.argv) == 5:
        batch_size = 64
        global img_height
        global img_width
        input_path = sys.argv[1]
        img_height = int(sys.argv[2])
        img_width = int(sys.argv[3])
        normalise = sys.argv[4]
        if normalise == "Y":
            stain_normalisation = True

        input_path = PathHandler(input_path)

        if input_path.valid:
            # Step 1. load the model that has been trained from a checkpoint file

            # Adapted from https://www.tensorflow.org/tutorials/keras/save_and_load
            # A lot of the code here is duplicated to allow the model to be
            # recreated. This could be improved by creating a create_model()
            # function in another file.

            AUTOTUNE = tf.data.AUTOTUNE
            normalization_layer = layers.Rescaling(1. / 255)
            num_classes = 2
            # Create a basic model instance
            global model
            from models.second_model import MakeModel
            model = MakeModel(img_height, img_width, num_classes)
            model.compile(optimizer='Adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(
                              from_logits=True),
                          metrics=['accuracy'])

            model.build((None, img_height, img_width, 3))
            # Load the weights from pre-trained model
            model_name = "beta_a_n"
            print("Using model: {}".format(model_name))
            checkpoint_path = "E:/fyp/models/{}/cp.ckpt".format(model_name)
            try:
                model.load_weights(checkpoint_path)
            except:
                print("Model not loaded correctly")
                exit()
            # Step 2. load the image to check

            # Process arguments

            provided_path = sys.argv[1]
            path = PathHandler(provided_path)

            pipeline = ImagePipeline()

            if path.folder():
                # The tail of the provided path is a folder
                # This will only be the case if all files are being
                # classified within the folder
                pipeline = ImagePipeline()
                classify_directory(pipeline, path)
            # if os.path.isfile(os.path.join(interpreted_path[0],interpreted_path[1])):
            elif path.file():
                # The tail of the provided path is a file
                # This will only be the case if a specific filed
                # is being classified
                if config['patch'] == "True":
                    try:
                        print(path.provided_path)
                        slide = openslide.OpenSlide(
                            # "E:\\Data\\Positive\\22113.svs")
                            path.provided_path)
                        # layer = 1  # 1/4 in H&N Data
                        layer = 2  # 1/16 in H&N Data
                        # layer = 4  # 1/16 in Cam16 Data
                        mask = Image.new(mode="RGB", size=(
                            slide.level_dimensions[layer][0] // 100, slide.level_dimensions[layer][1] // 100))
                        pixel_map = mask.load()
                        ratio = slide.level_dimensions[0][0] // slide.level_dimensions[layer][0]
                        # Iterate over the center point of every 100x100 region of the slide
                        for i in range(0, slide.level_dimensions[0][0] - 99 * ratio, 100 * ratio):
                            for j in range(0, slide.level_dimensions[0][1] - 99 * ratio, 100 * ratio):
                                try:
                                    tile = slide.read_region(
                                        (i, j), layer, (100, 100)).convert("RGB")
                                    # Check if patch is background (Section 4.3)
                                    min_r = 255
                                    min_g = 255
                                    min_b = 255
                                    for x in range(0, tile.width):
                                        for y in range(0, tile.height):
                                            pixel = tile.getpixel(
                                                (x, y))
                                            if pixel[0] < min_r:
                                                min_r = pixel[0]
                                            if pixel[1] < min_g:
                                                min_g = pixel[1]
                                            if pixel[2] < min_b:
                                                min_b = pixel[2]
                                    threshold = 200
                                    if min_r >= threshold and min_g >= threshold and min_b >= threshold:
                                        color = (0, 0, 0)
                                    else:
                                        color = classify_array(
                                            tile)
                                except:
                                    color = (0, 0, 0)
                                pixel_map[i // (100 * ratio), j //
                                          (100 * ratio)] = color
                            mask.save("mask.png")
                            print("Column {}/{} complete".format(i // (100 * ratio) + 1,
                                                                 slide.level_dimensions[layer][0] // 100))
                        # mask.show()
                    except:
                        print(
                            "Unable to read slide due to OpenSlide issue")
                else:
                    pipeline = ImagePipeline()
                    print("A")
                    pipeline.new_path(os.path.join(path.dir, path.file_name))
                    print("B")
                    classify_image(pipeline, image_index)
                    print("C")
        else:
            print("Invalid input path")
            exit()
    else:
        print("Wrong arguments")
        exit()


if __name__ == "__main__":
    main()
