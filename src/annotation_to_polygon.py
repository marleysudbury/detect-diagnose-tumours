# Convert annotation file to a shapely polygon and output patches according to
# their annotation.

# Written by Marley Sudbury (1838838)
# for CM3203 One Semester Individual Project

import os
import time
import json
import numpy as np
from shapely.geometry import Point
from shapely.geometry import shape
from shapely.geometry.polygon import Polygon
from PIL import Image
from utils.normalise_staining import normalizeStaining
from utils.load_config import config
openslidehome = config['openslide_path']

os.add_dll_directory(openslidehome)
import openslide

# Parameters to run the program
# TODO: take these from command line
filename = "22076"
slide_classification = "Negative"
normalise = False

with open('D:\\fyp\\Training Data !\\Head_Neck_Annotations\\{}\\{}.geojson'.format(slide_classification, filename)) as f:
    json_data = json.load(f)

regions = []
contains_negative = False
contains_positive = False

for region in json_data['features']:
    json_coords = region['geometry']['coordinates'][0]
    print(len(json_coords))
    if (len(json_coords) == 1):
         json_coords = json_coords[0]
    polygon = Polygon(json_coords)
    name = region['properties']['classification']['name']
    regions.append([polygon, name])
    if name == "Negative":
        contains_negative = True
    elif name == "Positive":
        contains_positive = True


print(regions)

class_count = {
    'Negative': 0,
    'Positive': 0,
    'Other': 0
}

slide = openslide.OpenSlide(
    "E:\\Data\\{}\\{}.svs".format(slide_classification, filename))

print(slide.dimensions)

layer = 2 # 1/16
ratio = slide.level_dimensions[0][0] // slide.level_dimensions[layer][0]

# Iterate over the center point of every 100x100 region of the slide
for i in range(0, slide.level_dimensions[0][0] - 99 * ratio, 100 * ratio):
    for j in range(0, slide.level_dimensions[0][1] - 99 * ratio, 100 * ratio):
        # point = Point(i+(50*ratio), j+(50*ratio))
        patch_polygon = Polygon([[i, j], [i+100* ratio, j], [i+100* ratio, j+100* ratio], [i, j+100* ratio]])
        patch_class = "Other"
        for region in regions:
            if patch_polygon.within(region[0]) and patch_class != "Positive":
                # Order or priority: Positive > Negative > Other. I.e.,:
                # - If the patch contains neither Positive nor Negative, it is Other
                # - If it contains Negative but not Positive, it is Negative
                # - If it contains Positive, it is Positive
                patch_class = region[1]

        # if patch_class == "Negative" and class_count[patch_class] < 250 or class_count[patch_class] < 100:
        if class_count[patch_class] < 1000 and patch_class == "Negative":
            try:
                tile = slide.read_region((i, j), layer, (100, 100)).convert("RGB")
                if normalise:
                    tile = np.transpose(np.array(tile), axes=[1,0,2])
                    tile = normalizeStaining(img=tile)[0]
                    Image.fromarray(tile).save('D:\\fyp\\Training Data !\\Head_Neck_Patch_1-16\\{}\\{}_{}_{}.png'.format(
                        patch_class, filename, i, j))
                else:
                    tile.save('D:\\fyp\\Training Data !\\Head_Neck_Patch_1-16\\{}\\{}_{}_{}.png'.format(
                        patch_class, filename, i, j))
                class_count[patch_class] += 1
                print("This patch: {}. P:{}/100 N:{}/250 O:{}/100".format(patch_class,
                      class_count['Positive'], class_count['Negative'], class_count['Other']))
                # if class_count['Positive'] == class_count['Other'] == 100 and class_count['Negative'] == 250:
                #     quit()
                #
                # if class_count['Positive'] == class_count['Other'] == 100 and not contains_negative:
                #     quit()
                if class_count['Negative'] == 1000:
                    quit()
            except Exception as err:
                print("An error occured while reading the region")
                print("{}: {}".format(type(err).__name__, err))
