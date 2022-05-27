# Image processing pipeline to take microscope images
# (multi-page .tif or .svs files) and use them for training
# or classification in Tensorflow

# Written by Marley Sudbury (1838838)
# for CM3203 One Semester Individual Project

import numpy as np
import math
import os
import cv2
import sys
from .path_handler import PathHandler
from .normalise_staining import normalizeStaining
from PIL import Image

# These files are required, they can be downloaded at:
# https://github.com/libvips/libvips/releases
# Change this for your install location and vips version, and remember to
# use double backslashes
from .load_config import config
# vipshome = config['libvips_path']

# # Include it in path PATH
# os.environ['PATH'] = vipshome + os.pathsep + os.environ['PATH']
# import pyvips
openslidehome = config['openslide_path']

os.add_dll_directory(openslidehome)
import openslide


class ImagePipeline:
    # This class handles image pipeline between .svs / .tif and Tensorflow
    def __init__(self, path=None):
        self.path = path

    def new_path(self, path):
        self.path = path

    def normalise_image(self, img):
        # Apply stain normalisation and
        # normalise the size of the images
        normalised_image = normalizeStaining(img=img)
        return normalised_image

    def squarify(self, M, val):
        # Adapted from https://stackoverflow.com/a/45989739
        (a, b, c) = M.shape
        if a > b:
            amount = math.floor((a - b) / 2)
            padding = ((0, 0), (amount, amount), (0, 0))
        else:
            amount = math.floor((b - a) / 2)
            padding = ((amount, amount), (0, 0), (0, 0))
        return np.pad(M, padding, mode='constant', constant_values=val)

    def convert_image(self, index, image_height=None, image_width=None, square=False):
        # Returns a numpy array of the specified image
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

        try:
            img = openslide.OpenSlide(self.path).get_thumbnail((100, 100))
            img.show()
            # if image_height is None:
            #     image_height = img.height
            # if image_width is None:
            #     image_width = img.width

            # img_array = np.ndarray(
            #     buffer=img.write_to_memory(),
            #     dtype=format_to_dtype[img.format],
            #     shape=[img.height, img.width, img.bands]
            # )

            if square:
                img = self.squarify(np.asarray(img), 255)
            img = cv2.resize(img, dsize=(
                image_height, image_width), interpolation=cv2.INTER_NEAREST)
            img = Image.fromarray(img)
            img.show()
            return img
        except Exception as err:
            print("An error occured while reading the image")
            print("{}: {}".format(type(err).__name__, err))

    def convert_batch(self):
        # Shouldn't be required
        # Instead, iterate_files() in path_handler
        # and convert_image() on each
        pass


if __name__ == "__main__":
    # python image_pipeline.py source(path) destination(path) height(int) width(int) normalisation(Y/N)
    # if source is a directory, convert all files within
    # if source is a single image, convert that images
    # save resulting files in destination

    # TODO: make some arguments optional
    # TODO: consider squarify argument
    # TODO: consider index argument
    if len(sys.argv) == 6:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        height = int(sys.argv[3])
        width = int(sys.argv[4])
        normalise = sys.argv[5]

        input_path = PathHandler(input_path)
        output_path = PathHandler(output_path)

        if input_path.valid and output_path.valid:

            pipeline = ImagePipeline()

            dtype_to_format = {
                'uint8': 'uchar',
                'int8': 'char',
                'uint16': 'ushort',
                'int16': 'short',
                'uint32': 'uint',
                'int32': 'int',
                'float32': 'float',
                'float64': 'double',
                'complex64': 'complex',
                'complex128': 'dpcomplex',
            }

            # TODO: ensure that the provided output_path matches with the provided input_path
            if input_path.folder():
                # Iterate over each image in the folder
                for file in input_path.iterate_files():
                    try:
                        # Convert single image
                        pipeline.new_path(file)
                        array = pipeline.convert_image(
                            3, image_height=height, image_width=width, square=True)
                        if normalise == "Y":
                            array = pipeline.normalise_image(array)[0]
                        else:
                            if normalise != "N":
                                print(
                                    "Invalid argument for normalise. Please use Y or N. Running with default value of N.")
                            else:
                                pass
                        image = Image.fromarray(array)
                        file_name = file.split(
                            os.path.sep)[-1].split(".")[0] + ".png"
                        image.write_to_file(os.path.join(
                            output_path.dir, file_name))

                    except Exception as err:
                        print("An error occured while converting the images")
                        print("{}: {}".format(type(err).__name__, err))
            elif input_path.file():
                try:
                    # Convert single image
                    pipeline.new_path(os.path.join(
                        input_path.dir, input_path.file_name))
                    array = pipeline.convert_image(
                        3, image_height=height, image_width=width, square=True)
                    if normalise == "Y":
                        array = pipeline.normalise_image(array)[0]
                    else:
                        if normalise != "N":
                            print(
                                "Invalid argument for normalise. Please use Y or N. Running with default value of N.")
                        else:
                            pass
                    image = Image.fromarray(array)
                    file_name = input_path.file_name.split(".")[0] + ".png"
                    image.write_to_file(os.path.join(
                        output_path.dir, file_name))

                except Exception as err:
                    print("An error occured while converting the image")
                    print("{}: {}".format(type(err).__name__, err))
            else:
                # Invalid path
                print("Provided input path is not valid.")

    else:
        print("Usage: python image_pipeline.py source(path) destination(path) height(int) width(int) normalisation(Y/N)")
