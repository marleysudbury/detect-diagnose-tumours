# https://github.com/One-sixth/imagescope_xml_utils

from imagescope_xml_utils.imagescope_xml_utils import ImageScopeXmlReader, ImageScopeXmlWriter
import numpy as np
import json

print("Hello world")

reader = ImageScopeXmlReader("22063.xml")

print(len(reader.get_contours()[0][0]))

min_x = 9999999999999999
min_y = 9999999999999999
max_x = 0
max_y = 0

for coord in reader.get_contours()[0][0]:
    if coord[1] < min_x:
        min_x = coord[1]
    if coord[0] < min_y:
        min_y = coord[0]
    if coord[1] > max_x:
        max_x = coord[1]
    if coord[0] > max_y:
        max_y = coord[0]

print("Min X: {}, Min Y: {}, Max X: {}, Max Y: {}".format(min_x, min_y, max_x, max_y))

writer = ImageScopeXmlWriter()

box = ([np.array([[min_y, min_x], [min_y, max_x], [max_y, max_x], [max_y, min_x]])],[(255,0,0)])

writer.add_boxes(box[0], box[1])
writer.add_contours(reader.get_contours()[0], reader.get_contours()[1])

with open('22063.geojson') as f:
    json_data = json.load(f)

json_coords = np.array(json_data['features'][0]['geometry']['coordinates'])
for pair in json_coords[0]:
    temp = pair[0]
    pair[0] = pair[1]
    pair[1] = temp
    
json_colour = [(25, 200, 60)]

writer.add_contours(json_coords, json_colour)

writer.write("square.xml")