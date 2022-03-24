# Convert annotation file to a shapely polygon

# Written my Marley Sudbury (1838838)
# for CM3203 One Semester Individual Project

# from imagescope_xml_utils.imagescope_xml_utils import ImageScopeXmlReader
#
# reader = ImageScopeXmlReader("22063.xml")

import json
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

with open('22063.geojson') as f:
    json_data = json.load(f)

regions = []

for region in json_data['features']:
    json_coords = region['geometry']['coordinates'][0]
    polygon = Polygon(json_coords)
    regions.append([polygon, region['properties']['classification']['name']])

print(regions)
