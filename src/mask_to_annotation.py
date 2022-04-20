# Takes a .PNG pixel mask and generats a
# GeoJSON annotation for QuPath

# Written by Marley Sudbury (1838838)
# for CM3203 One Semester Individual Project

from PIL import Image
import json

SCALE = 100  # How much bigger the WSI is compared the mask

MASK_FILE = "mask.png"
MASK_IMAGE = Image(MASK_FILE)
print(MASK_IMAGE)
