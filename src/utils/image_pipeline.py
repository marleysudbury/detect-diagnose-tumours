# Image processing pipeline to take microscope images
# (multi-page .tif or .svs files) and use them for training
# or classification in Tensorflow

# Written my Marley Sudbury (1838838)
# for CM3203 One Semester Individual Project

import numpy as np
import math
import os
import cv2
import sys
from utils.path_handler import PathHandler

# These files are required, they can be downloaded at:
# https://github.com/libvips/libvips/releases
# Change this for your install location and vips version, and remember to
# use double backslashes
from utils.load_config import config
vipshome = config['libvips_path']

# Include it in path PATH
os.environ['PATH'] = vipshome + os.path.sep + os.environ['PATH']

import pyvips


class ImagePipeline:
    # This class handles image pipeline between .svs / .tif and Tensorflow
    def ImagePipeline(self, path=None):
        self.path = path

    def new_path(self, path):
        self.path = path

    def normalise_image(self):
        # Apply stain normalisation and
        # normalise the size of the images
        pass

    def squarify(M, val):
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
            img = pyvips.Image.tiffload(self.path, page=index)

            if image_height is None:
                image_height = img.height
            if image_width is None:
                image_width = img.width

            img_array = np.ndarray(
                buffer=img.write_to_memory(),
                dtype=format_to_dtype[img.format],
                shape=[img.height, img.width, img.bands]
            )

            img_array = cv2.resize(img_array, dsize=(
                image_height, image_width), interpolation=cv2.INTER_CUBIC)
            if square:
                img_array = self.squarify(img_array, 255)
            return img_array
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
        input_path  = sys.argv[1]
        output_path = sys.argv[2]
        height      = sys.argv[3]
        width       = sys.argv[4]
        normalise   = sys.argv[5]

        input_path = PathHandler(input_path)
        output_path = PathHandler(output_path)

        pipeline = ImagePipeline()

        # TODO: ensure that the provided output_path matches with the provided input_path
        if input_path.folder():
            # Iterate over each image in the folder
            for file in input_path.iterate_files():
                # Convert single image
                pipeline.new_path(file)
                array = pipeline.convert_image(2, image_height=height, image_width=width, square=True)
                if normalise == "Y":
                    pass
                else:
                    if normalise != "N":
                        print("Invalid argument for normalise. Please use Y or N. Running with default value of N.")
                    else:
                        pass
                height, width, bands = array.shape
                linear = array.reshape(width * height * bands)
                image = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                                dtype_to_format[str(new_array.dtype)])
                image = image.thumbnail_image(width, height=height, crop=True)
                file_name = file.split("/")[-1].split(".")[0] + ".png"
                image.write_to_file(os.path.join(output_path.dir, file_name))
        elif input_path.file():
            # Convert single image
            pipeline.new_path(os.path.join(input_path.dir, input_path.file_name))
            array = pipeline.convert_image(2, image_height=height, image_width=width, square=True)
            if normalise == "Y":
                pass
            else:
                if normalise != "N":
                    print("Invalid argument for normalise. Please use Y or N. Running with default value of N.")
                else:
                    pass
            height, width, bands = array.shape
            linear = array.reshape(width * height * bands)
            image = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                            dtype_to_format[str(new_array.dtype)])
            image = image.thumbnail_image(width, height=height, crop=True)
            image.write_to_file(os.path.join(output_path.dir, output_path.file))
        else:
            # Invalid path
            print("Provided input path is not valid.")
    else:
        print("Usage: python image_pipeline.py source(path) destination(path) height(int) width(int) normalisation(Y/N)")
