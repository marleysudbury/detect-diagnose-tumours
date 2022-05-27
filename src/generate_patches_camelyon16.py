# Code to generate patches from slides and annotations in the Camelyon16 training dataset
# This is separate to the code for generating patches for Head and Neck 5000 dataset
# because different processes are used. See report section 4.3

# Written by Marley Sudbury (1838838)
# for CM3203 One Semester Individual Project

import os
import time
import json
import numpy as np
import random
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

list_of_filenames = [
    "tumor_001",
    "tumor_002",
    "tumor_003",
    "tumor_004",
    "tumor_005",
    "tumor_006",
    "tumor_007",
    "tumor_008",
    "tumor_009",
    "tumor_010"
]

for filename in list_of_filenames:
    for normalise in [False]:
        with open('E:\\fyp\\Training Data !\\Cam16Annotations\\GeoJSON\\{}.geojson'.format(filename)) as f:
            json_data = json.load(f)
        regions = []
        for region in json_data['features']:
            json_coords = region['geometry']['coordinates'][0]
            print(len(json_coords))
            if (len(json_coords) == 1):
                json_coords = json_coords[0]
            polygon = Polygon([[float(item[0]), float(item[1])]
                              for item in json_coords])
            name = region['properties']['classification']['name']
            regions.append([polygon, name])

        print(regions)

        # Keeps track of negative regions randomly visited
        visited_regions = []

        # Keeps track of number of each class collected
        class_count = {
            'Negative': 0,
            'Positive': 0
        }

        slide = openslide.OpenSlide(
            "E:\\fyp\\Training Data !\\Cam16\\Training\\Tumor\\{}.tif".format(filename))

        print(slide.dimensions)

        layer = 4  # 1/16
        ratio = slide.level_dimensions[0][0] // slide.level_dimensions[layer][0]

        # Iterate over every 100x100 region of the slide to get all positive patches
        for i in range(0, slide.level_dimensions[0][0] - 99 * ratio, 100 * ratio):
            for j in range(0, slide.level_dimensions[0][1] - 99 * ratio, 100 * ratio):
                patch_polygon = Polygon(
                    [[i, j], [i + 100 * ratio, j], [i + 100 * ratio, j + 100 * ratio], [i, j + 100 * ratio]])
                patch_class = "Negative"
                for region in regions:
                    if patch_polygon.within(region[0]):
                        patch_class = "Positive"

                        try:
                            tile = slide.read_region(
                                (i, j), layer, (100, 100)).convert("RGB")
                            if normalise:
                                tile = np.transpose(
                                    np.array(tile), axes=[1, 0, 2])
                                tile = normalizeStaining(img=tile)[0]
                                Image.fromarray(tile).save('E:\\fyp\\Training Data !\\Cam16PatchNormal_1-16\\Training\\Positive\\{}_{}_{}.png'.format(
                                    filename, i, j))
                            else:
                                tile.save('E:\\fyp\\Training Data !\\Cam16Patch_1-16\\Training\\Positive\\{}_{}_{}.png'.format(
                                    filename, i, j))
                            class_count[patch_class] += 1
                            print("This patch: {}. P:{}, N:{}".format(patch_class,
                                                                      class_count['Positive'], class_count['Negative']))
                        except Exception as err:
                            print("An error occured while reading the region")
                            print("{}: {}".format(type(err).__name__, err))

        skip_slide = False
        # Iterate through patches until negative patches equal positive
        for i in range(int((slide.level_dimensions[0][0] - 99 * ratio) / 2), slide.level_dimensions[0][0] - 99 * ratio, 100 * ratio):
            for j in range(int((slide.level_dimensions[0][1] - 99 * ratio) / 2), slide.level_dimensions[0][1] - 99 * ratio, 100 * ratio):
                if class_count["Negative"] >= class_count["Positive"]:
                    skip_slide = True
                if skip_slide:
                    break
                patch_class = "Negative"
                # Check if it is positive
                patch_polygon = Polygon(
                    [[i, j], [i + 100 * ratio, j], [i + 100 * ratio, j + 100 * ratio], [i, j + 100 * ratio]])
                for region in regions:
                    if patch_polygon.within(region[0]):
                        patch_class = "Positive"
                if patch_class == "Negative":
                    try:
                        tile = slide.read_region(
                            (i, j), layer, (100, 100)).convert("RGB")
                        # Make sure it's not over the RGB threshold
                        # Check if patch is background (Section 4.3)
                        min_r = 255
                        min_g = 255
                        min_b = 255
                        for x in range(0, tile.width):
                            for y in range(0, tile.height):
                                pixel = tile.getpixel((x, y))
                                if pixel[0] < min_r:
                                    min_r = pixel[0]
                                if pixel[1] < min_g:
                                    min_g = pixel[1]
                                if pixel[2] < min_b:
                                    min_b = pixel[2]
                        threshold = 220
                        if min_r >= threshold and min_g >= threshold and min_b >= threshold:
                            # print("Background")
                            pass
                        else:
                            if normalise:
                                tile = np.transpose(
                                    np.array(tile), axes=[1, 0, 2])
                                tile = normalizeStaining(img=tile)[0]
                                Image.fromarray(tile).save('E:\\fyp\\Training Data !\\Cam16PatchNormal_1-16\\Training\\Negative\\{}_{}_{}.png'.format(
                                    filename, i, j))
                            else:
                                tile.save('E:\\fyp\\Training Data !\\Cam16Patch_1-16\\Training\\Negative\\{}_{}_{}.png'.format(
                                    filename, i, j))
                            class_count[patch_class] += 1
                            print("This patch: {}. P:{}, N:{}".format(patch_class,
                                                                      class_count['Positive'], class_count['Negative']))
                    except Exception as err:
                        print("An error occured while reading the region")
                        print("{}: {}".format(type(err).__name__, err))
