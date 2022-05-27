# Normalise pixel colour values to between 0 and 255
# So they are more visible in report

from PIL import Image
from numpy import interp
path = "C:\\Users\\Marley\\Downloads\\confidence_mask.png"

mask = Image.open(path)
mask_pixel_map = mask.load()

for x in range(mask.width):
    for y in range(mask.height):
        pixel = mask_pixel_map[x, y]
        red = pixel[0]
        green = pixel[1]
        if red == green == 0:
            pass
        else:
            red = int(interp(red, [0, 100], [0, 255]))
            green = int(interp(green, [0, 100], [0, 255]))
            pixel = (red, green, 0)
            mask_pixel_map[x, y] = pixel

mask.save(path)
mask.show()
