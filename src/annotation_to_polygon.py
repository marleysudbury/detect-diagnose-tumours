# Convert annotation file to a shapely polygon

# Written my Marley Sudbury (1838838)
# for CM3203 One Semester Individual Project

import os
import time
import json
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from PIL import Image
from utils.load_config import config
openslidehome = config['openslide_path']

os.add_dll_directory(openslidehome)
import openslide

filename = "22158"

with open('D:\\Training Data !\\Head_Neck_Annotations\\{}.geojson'.format(filename)) as f:
    json_data = json.load(f)

regions = []
contains_negative = False

for region in json_data['features']:
    json_coords = region['geometry']['coordinates'][0]
    polygon = Polygon(json_coords)
    name = region['properties']['classification']['name']
    regions.append([polygon, name])
    if name == "Negative":
        contains_negative = True


print(regions)

class_count = {
    'Negative': 0,
    'Positive': 0,
    'Other': 0
}

slide = openslide.OpenSlide(
    "E:\\Data\\Positive\\{}.svs".format(filename))

print(slide.dimensions)

layer = 2 # 1/16
ratio = slide.level_dimensions[0][0] // slide.level_dimensions[layer][0]

# Iterate over the center point of every 100x100 region of the slide
for i in range(0, slide.level_dimensions[0][0] - 99 * ratio, 100 * ratio):
    for j in range(0, slide.level_dimensions[0][1] - 99 * ratio, 100 * ratio):
        point = Point(i+(50*ratio), j+(50*ratio))
        patch_class = "Other"
        for region in regions:
            if region[0].contains(point):
                patch_class = region[1]

        if patch_class == "Negative" and class_count[patch_class] < 500 or class_count[patch_class] < 200:
            try:
                tile = slide.read_region((i, j), layer, (100, 100))
                tile.save('D:\\Training Data !\\Head_Neck_Patch\\{}\\{}_{}_{}.png'.format(
                    patch_class, filename, i, j))
                class_count[patch_class] += 1
                print("This patch: {}. P:{}/200 N:{}/500 O:{}/200".format(patch_class,
                      class_count['Positive'], class_count['Negative'], class_count['Other']))
                if class_count['Positive'] == class_count['Other'] == 100 and class_count['Negative'] == 250:
                    quit()

                if class_count['Positive'] == class_count['Other'] == 100 and not contains_negative:
                    quit()
            except Exception as err:
                print("An error occured while reading the region")
                print("{}: {}".format(type(err).__name__, err))
