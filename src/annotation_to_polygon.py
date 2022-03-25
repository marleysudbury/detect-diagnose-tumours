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

with open('tumor_001.geojson') as f:
    json_data = json.load(f)

regions = []

for region in json_data['features']:
    json_coords = region['geometry']['coordinates'][0]
    polygon = Polygon(json_coords)
    regions.append([polygon, region['properties']['classification']['name']])


print(regions)

slide = openslide.OpenSlide(
    "E:\\Training Data !\\Cam16\\Training\\Tumor\\tumor_001.tif")

print(slide.dimensions)

# Iterate over the center point of every 100x100 region of the slide
for i in range(50, slide.dimensions[0], 100):
    for j in range(50, slide.dimensions[1], 100):
        point = Point(i, j)
        for region in regions:
            if region[0].contains(point):
                tile = slide.read_region((i - 50, j - 50), 0, (100, 100))
                tile.save('E:\\Training Data !\\Cam16Patch\\Training\\{}\\{}_{}_{}.png'.format(
                    region[1], "tumor_001", i, j))
