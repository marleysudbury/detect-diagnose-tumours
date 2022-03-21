# Image processing pipeline to take microscope images
# (multi-page .tif or .svs files) and use them for training
# or classification in Tensorflow

# Written my Marley Sudbury (1838838)
# for CM3203 One Semester Individual Project

import os

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
            img = pyvips.Image.tiffload(path, page=index)

            if image_height is None:
                image_height = img.height
            if image_width is None:
                image_width = img.width

            img_array = np.ndarray(
                buffer=img.write_to_memory(),
                dtype=format_to_dtype[img.format],
                shape=[img.height, img.width, img.bands]
            )

            img_array = cv2.resize(img_array, dsize=(img_height, img_width), interpolation=cv2.INTER_CUBIC)
            if square:
                img_array = squarify(img_array, 255)
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
    pass
