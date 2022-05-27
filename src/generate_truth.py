# Generates a ground truth mask PNG from annotation (GeoJSON)

# Written by Marley Sudbury (1838838)
# for CM3203 One Semester Individual Project
import os
from utils.load_config import config
openslidehome = config['openslide_path']

os.add_dll_directory(openslidehome)
import openslide
from PIL import Image
from shapely.geometry.polygon import Polygon
import json

# list_of_filenames = [
#     "tumor_001",
#     "tumor_002",
#     "tumor_003",
#     "tumor_004",
#     "tumor_005",
#     "tumor_006",
#     "tumor_007",
#     "tumor_008",
#     "tumor_009",
#     "tumor_010"
# ]

list_of_filenames = [
    "22070",
    "22071",
    "22081",
    "22082",
    "22083",
    # "22111",
    "22112",
    "22113",
    "22114",
    "22158"
]

for filename in list_of_filenames:

    # Location of the annotation
    input_name = "E:\\fyp\\Training Data !\\Head_Neck_Annotations\\Positive\\{}.geojson".format(
        filename)
    # input_name = "E:\\fyp\\Training Data !\\Cam16Annotations\\GeoJSON\\{}.geojson".format(
    # filename)

    # File that will be created
    output_name = "E:\\fyp\\Training Data !\\Head_Neck_Annotations\\PNG\\{}.png".format(
        filename)
    # output_name = "E:\\fyp\\Training Data !\\Cam16Annotations\\PNG\\{}.png".format(
    #     filename)

    # The slide is required to get the dimensions
    slide_name = "G:\\Data\\Positive\\{}.svs".format(filename)
    # slide_name = "E:\\fyp\\Training Data !\\Cam16\\Training\\Tumor\\{}.tif".format(
    #     filename)

    # Open the slide
    slide = openslide.OpenSlide(slide_name)

    layer = 2  # 1/16

    ratio = slide.level_dimensions[0][0] // slide.level_dimensions[layer][0]

    mask = Image.new(mode="RGB", size=(
        slide.level_dimensions[layer][0] // 100, slide.level_dimensions[layer][1] // 100))
    pixel_map = mask.load()

    with open(input_name) as f:
        json_data = json.load(f)
        regions = []
        for region in json_data['features']:
            json_coords = region['geometry']['coordinates'][0]
            if (len(json_coords) == 1):
                json_coords = json_coords[0]
            polygon = Polygon([[float(item[0]), float(item[1])]
                               for item in json_coords])
            name = region['properties']['classification']['name']
            if name != "Negative":
                regions.append([polygon, name])

    # Iterate over every pixel of every patch of the slide
    for x in range(mask.width):
        for y in range(mask.height):
            x_exp = (x * 100) * ratio
            y_exp = (y * 100) * ratio
            patch_polygon = Polygon([[x_exp, y_exp], [x_exp + 100 * ratio, y_exp], [
                                    x_exp + 100 * ratio, y_exp + 100 * ratio], [x_exp, y_exp + 100 * ratio]])
            colour = (100, 0, 0)
            for region in regions:
                if patch_polygon.intersects(region[0]):
                    colour = (0, 100, 0)
            pixel_map[x, y] = colour

    mask.save(output_name)
