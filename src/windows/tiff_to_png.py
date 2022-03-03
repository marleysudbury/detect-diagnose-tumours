# It should be trivial to convert from Tiff to PNG
# So here we go

# Adapted from https://stackoverflow.com/questions/62629946/python-converting-images-in-tif-format-to-png

import os

# change this for your install location and vips version, and remember to
# use double backslashes
vipshome = 'C:\\Users\\c1838838\\Downloads\\vips-dev-8.12\\bin'

# set PATH
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']

import pyvips

# image = pyvips.Image.tiffload("to_convert/normal_001.tif", page=2)
#
# image.write_to_file('fred.jpg[Q=50]')

# yourpath = os.path.basename("/Documents/Project/to_convert")
# destination = os.path.basename("/Documents/Project/converted")

yourpath = os.path.dirname("E:\\Data\\Negative\\")
destination = os.path.dirname("D:\\Training Data !\\Adam compressed\\Negative\\")

if os.path.isdir(yourpath):
    print("True")
else:
    print("False")

for root, dirs, files in os.walk(yourpath, topdown=False):
    print("test")
    for name in files:
        print(os.path.join(root, name))
        if os.path.splitext(os.path.join(root, name))[1].lower() == ".svs":
            if os.path.isfile(os.path.splitext(os.path.join(destination, name))[0] + ".jpg"):
                print ("A jpg file already exists for %s" % name)
            # If a PNG is *NOT* present, create one from the tiff.
            else:
                outfile = os.path.splitext(os.path.join(destination, name))[0] + ".jpg"
                try:
                    # im = Image.open(os.path.join(root, name))
                    print("Generating jpeg for %s" % name)
                    # im.thumbnail(im.size)
                    # im.save(outfile, "jpg", quality=100)
                    image = pyvips.Image.tiffload(os.path.join(root, name), page=1)

                    image.write_to_file(outfile,Q=50)
                except Exception as e:
                    print (e)
