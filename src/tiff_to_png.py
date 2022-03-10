# Image processing pipeline to take microscope images
# (multi-page .tif or .svs files) and use them for training
# or classification in Tensorflow

# Written my Marley Sudbury (1838838)
# for CM3203 One Semester Individual Project

import os
import numpy as np
import math

# These files are required, they can be downloaded at:
# https://github.com/libvips/libvips/releases
# Change this for your install location and vips version, and remember to
# use double backslashes
# vipshome = 'C:\\Users\\Marley\\Downloads\\vips-dev-8.12\\bin'
vipshome = 'C:\\Users\\c1838838\\Downloads\\vips-dev-8.12\\bin'

# Include it in path PATH
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
# os.environ['PATH'] = openslidehome + ';' + os.environ['PATH']

import pyvips
# import openslide

# Adapted from https://stackoverflow.com/questions/62629946/python-converting-images-in-tif-format-to-png
# Take images from this directory
yourpath = os.path.dirname("E:\\Data\\Negative\\")
# Save the images to this directory
destination = os.path.dirname("D:\\Training Data !\\Adam compressed\\Negative\\")

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

for root, dirs, files in os.walk(yourpath, topdown=False):
    for name in files:
        print(os.path.join(root, name))
        if os.path.splitext(os.path.join(root, name))[1].lower() == ".svs":
            if os.path.isfile(os.path.splitext(os.path.join(destination, name))[0] + ".png"):
                print ("A png file already exists for %s" % name)
            # If a PNG is *NOT* present, create one from the TIF.
            else:
                outfile = os.path.splitext(os.path.join(destination, name))[0] + ".png"
                try:
                    print("Generating png for %s" % name)
                    image = pyvips.Image.tiffload(os.path.join(root, name), page=2)

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


                    img_array = np.ndarray(
                        buffer=image.write_to_memory(),
                        dtype=format_to_dtype[image.format],
                        shape=[image.height, image.width, image.bands]
                    )
                    new_array = squarify(img_array, 255)
                    height, width, bands = new_array.shape
                    linear = new_array.reshape(width * height * bands)
                    image = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                                    dtype_to_format[str(new_array.dtype)])
                    image = image.thumbnail_image(1000, height=1000, crop=True)
                    image.write_to_file(outfile)
                except Exception as e:
                    print ("ERROR:",e)
