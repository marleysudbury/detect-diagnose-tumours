# Takes a .PNG pixel mask and generats a
# GeoJSON annotation for QuPath

# Written by Marley Sudbury (1838838)
# for CM3203 One Semester Individual Project

from PIL import Image
import json

SCALE = 1603  # Pixels in original to 1 pixel in mask

MASK_FILE = "mask.png"
MASK_IMAGE = Image.open(MASK_FILE)

json_obj = {}
json_obj["type"] = "FeatureCollection"
json_obj["features"] = []

# Iterate over all pixels
for y in range(0, MASK_IMAGE.height):
    for x in range(0, MASK_IMAGE.width):
        # print(x, y)
        # print(MASK_IMAGE.getpixel((x, y)))
        colour = MASK_IMAGE.getpixel((x, y))
        if x != 0 and y != 0:
            if colour == (255, 0, 0):
                # Red, i.e. positive tissue
                # Test: add the four corners of the pixel as vertices of a polygon
                polygon = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[]]}, "properties": {
                    "object_type": "annotation", "classification": {"name": "Tumor", "colorRGB": -3670016}, "isLocked": 'false'}}
                polygon["geometry"]["coordinates"][0].append(
                    [x * SCALE, y * SCALE])
                polygon["geometry"]["coordinates"][0].append(
                    [(x + 1) * SCALE, y * SCALE])
                polygon["geometry"]["coordinates"][0].append(
                    [(x + 1) * SCALE, (y + 1) * SCALE])
                polygon["geometry"]["coordinates"][0].append(
                    [x * SCALE, (y + 1) * SCALE])
                # Qupath requires a closed polygon, so finish with the first point again
                polygon["geometry"]["coordinates"][0].append(
                    polygon["geometry"]["coordinates"][0][0])
                # Add annotation to JSON
                json_obj["features"].append(polygon)

                # if MASK_IMAGE.getpixel((x - 1, y)) != (255, 0, 0):
                #     if MASK_IMAGE.getpixel((x - 1, y - 1)) != (255, 0, 0):
                #         if MASK_IMAGE.getpixel((x, y - 1)) != (255, 0, 0):
                # add xy, xy+1, x+1y

with open('anno_from_mask.json', 'w') as jsonFile:
    json.dump(json_obj, jsonFile)
