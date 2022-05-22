# Converts XML annotations from Camelyon16 dataset into GeoJSON

# Written by Marley Sudbury (1838838)
# for CM3203 One Semester Individual Project

import numpy as np
import json
import xml.etree.ElementTree as ET

# Adapted from:
# https://docs.python.org/3/library/xml.etree.elementtree.html

tree = ET.parse("D:\\fyp\\Training Data !\\Cam16_Annotations\\tumor_010.xml")

root = tree.getroot()

# print(root[0][0][0][0].tag)

json_obj = {}
json_obj["type"] = "FeatureCollection"
json_obj["features"] = []
for annotation in root[0]:
    # print(annotation.tag)
    polygon = {"type":"Feature","geometry":{"type":"Polygon","coordinates":[[]]}, "properties":{"object_type":"annotation","classification":{"name":"Tumor","colorRGB":-3670016},"isLocked":'false'}}
    for coord_pair in annotation[0]:
        polygon["geometry"]["coordinates"][0].append([coord_pair.get('X'), coord_pair.get('Y')])
    # Qupath requires a closed polygon, so finish with the first point again
    polygon["geometry"]["coordinates"][0].append(polygon["geometry"]["coordinates"][0][0])
    # Add annotation to JSON
    json_obj["features"].append(polygon)

# print(json_obj)
with open('my_output.geojson','w') as jsonFile:
    json.dump(json_obj, jsonFile)

# reader = ImageScopeXmlReader("D:\\Training Data !\\Cam16_Annotations\\tumor_001.xml")

# print(len(reader.get_contours()[0][0]))
#
# min_x = 9999999999999999
# min_y = 9999999999999999
# max_x = 0
# max_y = 0
#
# for coord in reader.get_contours()[0][0]:
#     if coord[1] < min_x:
#         min_x = coord[1]
#     if coord[0] < min_y:
#         min_y = coord[0]
#     if coord[1] > max_x:
#         max_x = coord[1]
#     if coord[0] > max_y:
#         max_y = coord[0]
#
# print("Min X: {}, Min Y: {}, Max X: {}, Max Y: {}".format(min_x, min_y, max_x, max_y))
#
# writer = ImageScopeXmlWriter()
#
# box = ([np.array([[min_y, min_x], [min_y, max_x], [max_y, max_x], [max_y, min_x]])],[(255,0,0)])
#
# writer.add_boxes(box[0], box[1])
# writer.add_contours(reader.get_contours()[0], reader.get_contours()[1])
#
# with open('22063.geojson') as f:
#     json_data = json.load(f)
#
# json_coords = np.array(json_data['features'][0]['geometry']['coordinates'])
# for pair in json_coords[0]:
#     temp = pair[0]
#     pair[0] = pair[1]
#     pair[1] = temp
#
# color_int = json_data['features'][0]['properties']['classification']['colorRGB']
# color_bytes = color_int.to_bytes(4, byteorder='big', signed=True)
# # Alpha doesn't work because Python insists on signed=True
# # which would give a negative value as the int is stored
# # with two's complement
# r = color_bytes[1]
# g = color_bytes[2]
# b = color_bytes[3]
#
# json_colour = [(r, g, b)]
#
# # json_colour = [(25, 200, 60)]
#
# writer.add_contours(json_coords, json_colour)
#
# writer.write("square.xml")
