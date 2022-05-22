# Generates a ground truth mask PNG from annotation (GeoJSON)

# Written by Marley Sudbury (1838838)
# for CM3203 One Semester Individual Project

# Location of the annotation
input_name = "tumor_001.geojson"

# File that will be created
output_name = "tumor_001.png"

# The slide is required to get the dimensions
slide_name = "D:\\fyp\\Training Data !\\Cam16\\Training\\Tumor\\tumor_001.tif"

# Open the slide
slide = openslide.OpenSlide(slide_name)

layer = 2  # 1/16
mask = Image.new(mode="RGB", size=(
    slide.level_dimensions[layer][0] // 100, slide.level_dimensions[layer][1] // 100))
pixel_map = mask.load()

with open('D:\\fyp\\Training Data !\\Cam16_Annotations\\GeoJSON\\{}.geojson'.format(filename)) as f:
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


mask.save(output_name)
