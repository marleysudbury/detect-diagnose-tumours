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
vipshome = 'C:\\Users\\c1838838\\Downloads\\vips-dev-8.12\\bin'

# Include it in path PATH
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']

import pyvips

# Adapted from https://stackoverflow.com/questions/62629946/python-converting-images-in-tif-format-to-png
# Take images from this directory
yourpath = os.path.dirname("E:\\Data\\Negative\\")
# Save the images to this directory
destination = os.path.dirname("D:\\Training Data !\\Adam compressed\\Negative\\")

for root, dirs, files in os.walk(yourpath, topdown=False):
    for name in files:
        print(os.path.join(root, name))
        if os.path.splitext(os.path.join(root, name))[1].lower() == ".svs":
            if os.path.isfile(os.path.splitext(os.path.join(destination, name))[0] + ".jpg"):
                print ("A jpg file already exists for %s" % name)
            # If a JPG is *NOT* present, create one from the TIF.
            else:
                outfile = os.path.splitext(os.path.join(destination, name))[0] + ".jpg"
                try:
                    print("Generating jpeg for %s" % name)
                    image = pyvips.Image.tiffload(os.path.join(root, name), page=1)
                    image.write_to_file(outfile,Q=50)
                except Exception as e:
                    print (e)
