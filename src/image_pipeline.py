# Image processing pipeline to take microscope images
# (multi-page .tif or .svs files) and use them for training
# or classification in Tensorflow

# Written my Marley Sudbury (1838838)
# for CM3203 One Semester Individual Project

import os
from path_handler import PathHandler

# These files are required, they can be downloaded at:
# https://github.com/libvips/libvips/releases
# Change this for your install location and vips version, and remember to
# use double backslashes
vipshome = 'C:\\Users\\Marley\\Downloads\\vips-dev-8.12\\bin'

# Include it in path PATH
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']

import pyvips


class ImagePipeline:
    # This class handles image pipeline between .svs / .tif and Tensorflow
    def ImagePipeline():
        batch = False
        training = False
        origin_directory = None
        destination_directory = None

        path = PathHandler()

    def convert_image():
        pass

    def convert_batch():
        pass
