# Image processing pipeline to take microscope images
# (multi-page .tif or .svs files) and use them for training
# or classification in Tensorflow

# Written my Marley Sudbury (1838838)
# for CM3203 One Semester Individual Project

import pyvips
import os
from utils.path_handler import PathHandler


# These files are required, they can be downloaded at:
# https://github.com/libvips/libvips/releases
# Change this for your install location and vips version, and remember to
# use double backslashes
from utils.load_config import config
vipshome = config['libvips_path']

# Include it in path PATH
os.environ['PATH'] = vipshome + os.path.sep + os.environ['PATH']


class ImagePipeline:
    # This class handles image pipeline between .svs / .tif and Tensorflow
    def ImagePipeline():
        batch = False
        training = False
        origin_directory = None
        destination_directory = None

        path = PathHandler()

    def normalise_image():
        # Apply stain normalisation and
        # normalise the size of the images
        pass

    def convert_image():
        pass

    def convert_batch():
        # Shouldn't be required
        # Instead, iterate_files() in path_handler
        # and convert_image() on each
        pass


if __name__ == "__main__":
    pass
