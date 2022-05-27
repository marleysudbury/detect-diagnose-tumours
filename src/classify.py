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
import sys
from utils.normalise_staining import normalizeStaining
from utils.path_handler import PathHandler
from PIL import Image
# from utils.image_pipeline import ImagePipeline

# For Grad-CAM visualisation
# https://keras.io/examples/vision/grad_cam/

image_index = 1

normalise = True


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


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # https://keras.io/examples/vision/grad_cam/
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(
            last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def classify_array(array):
    # Used to classify patch
    # Takes an image and prints the classification and confidence
    # Red is negative
    # Green is positive
    try:
        class_names = ['Negative', 'Positive']
        if normalise:
            array = np.transpose(np.array(array), axes=[1, 0, 2])
            array = normalizeStaining(img=array)[0]
        img_array = tf.expand_dims(array, 0)  # Create a batch
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        # print(
        #     "This image most likely belongs to {} with a {:.2f} percent confidence."
        #     .format(class_names[np.argmax(score)], 100 * np.max(score))
        # )

        # if class_names[np.argmax(score)] == "Positive":
        #     return (int(score[0].numpy() * 100), int(score[1].numpy() * 100), 0)
        # elif class_names[np.argmax(score)] == "Negative":
        #     return (int(score[1].numpy() * 100), int(score[0].numpy() * 100), 0)

        return (int(score[0].numpy() * 100), int(score[1].numpy() * 100), 0)
    except Exception as err:
        print("An error occured while normalising the region")
        print("{}: {}".format(type(err).__name__, err))
        return (0, 0, 0)


def classify_image(pipeline, index):
    # Takes an image and prints the classification and confidence
    try:
        # img = pyvips.Image.tiffload(path, page=index)
        # img_array = np.ndarray(
        #     buffer=img.write_to_memory(),
        #     dtype=format_to_dtype[img.format],
        #     shape=[img.height, img.width, img.bands]
        # )
        # img_array = cv2.resize(img_array, dsize=(img_height, img_width), interpolation=cv2.INTER_CUBIC)
        # img_array = squarify(img_array, 255)
        # img_array = tf.image.resize(img_array, [img_height, img_width])

        img_array = pipeline.convert_image(index, img_height, img_width, True)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        class_names = ['Negative', 'Positive']
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
        # return

        model_builder = keras.applications.xception.Xception
        img_size = (img_height, img_width)
        preprocess_input = keras.applications.xception.preprocess_input
        decode_predictions = keras.applications.xception.decode_predictions

        last_conv_layer_name = "conv2d_2"

        # Prepare image
        # img_array = preprocess_input(get_img_array(img_path, size=img_size))

        # Make model
        # model = model_builder(weights="imagenet")

        # Remove last layer's softmax
        model.layers[-1].activation = None

        # Print what the top predicted class is
        # preds = model.predict(img_array)
        # print("Predicted:", decode_predictions(preds, top=1)[0])

        # Generate class activation heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        img = img_array[0]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[0], img.shape[1]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * 0.4 + img
        superimposed_img = keras.preprocessing.image.array_to_img(
            superimposed_img)

        # Save the superimposed image
        # superimposed_img.save(cam_path)

        # Display Grad CAM
        # display(Image(cam_path))

        # imposed = heatmap * 0.4 + img_array

        # Display heatmap
        plt.matshow(jet_heatmap)
        plt.show()
    except Exception as err:
        print("An error occured while classifying the image")
        print("{}: {}".format(type(err).__name__, err))


def classify_directory(pipeline, path):
    for file in path.iterate_files():
        pipeline.new_path(file)
        classify_image(pipeline, image_index)


def main():
    # Step 1. load the model that has been trained from a checkpoint file

    # Adapted from https://www.tensorflow.org/tutorials/keras/save_and_load
    # A lot of the code here is duplicated to allow the model to be
    # recreated. This could be improved by creating a create_model()
    # function in another file.

    batch_size = 64
    global img_height
    global img_width

    img_height = 100
    img_width = 100

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

    model.summary()

    # checkpoint_path = "E:/fyp/models/alpha_p_n/cp.ckpt"
    # model.load_weights(checkpoint_path)

    # Step 2. load the image to check

    # .tif file
    # img = tf.keras.utils.load_img(pathlib.Path('D:\\Training Data !\\Adam compressed\\Negative\\22073.jpg'), target_size = (180, 180))

    # Process arguments

    provided_path = sys.argv[1]
    path = PathHandler(provided_path)

    # pipeline = ImagePipeline()

    # interpreted_path = os.path.split(provided_path)
    # print(interpreted_path)
    # dir = None
    # file = None

    # if os.path.isdir(os.path.join(interpreted_path[0],interpreted_path[1],'\\')):
    if path.folder():
        # The tail of the provided path is a folder
        # This will only be the case if all files are being
        # classified within the folder
        # print("dir")
        # dir = os.path.join(interpreted_path[0], interpreted_path[1])
        pipeline = ImagePipeline()
        classify_directory(pipeline, path)
    # if os.path.isfile(os.path.join(interpreted_path[0],interpreted_path[1])):
    elif path.file():
        # The tail of the provided path is a file
        # This will only be the case if a specific filed
        # is being classified
        # print("file")
        # dir = interpreted_path[0]
        # file = interpreted_path[1]
        if config['patch'] == "True":
            list_of_files = [
                "22070",
                # "22071",
                # "22081",
                # "22082",
                # "22083",
                # # "22111",
                # "22112",
                # "22113",
                # "22114",
                # "22158"
            ]
            # list_of_models = [
            #     "beta_p",
            #     "beta_p_a",
            #     # "beta_p_n",
            #     # "beta_p_a_n"
            # ]
            list_of_models = [
                "beta_p_1-4",
                "beta_p_a_1-4",
                "beta_p_n_1-4",
                "beta_p_a_n_1-4"
            ]

            for model_name in list_of_models:
                checkpoint_path = "E:/fyp/models/{}/cp.ckpt".format(model_name)
                model.load_weights(checkpoint_path)
                for normal in [True]:
                    global normalise
                    normalise = normal
                    normal_name = "no_normal"
                    if normal:
                        normal_name = "normal"
                    for file_name in list_of_files:
                        try:
                            slide = openslide.OpenSlide(
                                # "E:\\Data\\Positive\\22113.svs")
                                "G:\\Data\\Positive\\{}.svs".format(file_name))
                            # print(slide.dimensions)
                            # print(slide.level_dimensions)
                            # mask = Image.new(mode = "RGB", size = (slide.dimensions[0]//100, slide.dimensions[1]//100))
                            # pixel_map = mask.load()
                            # # Iterate over the center point of every 100x100 region of the slide
                            # for i in range(0, slide.dimensions[0]-99, 100):
                            #     for j in range(0, slide.dimensions[1]-99, 100):
                            #         tile = slide.read_region(
                            #             (i, j), 0, (100, 100)).convert("RGB")
                            #         color = classify_array(tile)
                            #         pixel_map[i//100, j//100] = color
                            #     mask.save("mask.png")
                            # mask.show()
                            layer = 1  # 1/4 in H&N Data
                            # layer = 2  # 1/16 in H&N Data
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
                                                pixel = tile.getpixel((x, y))
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
                                            color = classify_array(tile)
                                    except:
                                        color = (0, 0, 0)
                                    pixel_map[i // (100 * ratio), j //
                                              (100 * ratio)] = color
                                mask.save("E:\\fyp\\predictions\\{}\\{}\\{}.png".format(
                                    model_name, normal_name, file_name))
                                print("Column {}/{} complete".format(i // (100 * ratio) + 1,
                                                                     slide.level_dimensions[layer][0] // 100))
                            # mask.show()
                        except:
                            print("Unable to read slide due to OpenSlide issue")
        else:
            pipeline = ImagePipeline()
            pipeline.new_path(os.path.join(path.dir, path.file))
            classify_image(pipeline, image_index)

    # if os.path.isdir(interpreted_path[0]):
    #     # The head of the provided path is a folder
    #     # This will only be the case if the specified image or
    #     # directory of images are in a directory other than the CWD
    #     # The provided path is a folder
    #     print("dir")
    #     dir = 0
    else:
        # The provided path is neither a file nor a folder
        print("The specified path is not valid :(")

    if '-a' in sys.argv:
        print('Classifying all images in directory')


if __name__ == "__main__":
    main()
