from uwimg import *

'''
# 1. Getting and setting pixels
im = load_image("data/dog.jpg")
for row in range(im.h):
    for col in range(im.w):
        set_pixel(im, 0, row, col, 0)
save_image(im, "dog_no_red")

# 3. Grayscale image
im = load_image("data/colorbar.png")
graybar = rgb_to_grayscale(im)
save_image(graybar, "graybar")

# 4. Shift Image
im = load_image("data/dog.jpg")
shift_image(im, 0, .4)
shift_image(im, 1, .4)
shift_image(im, 2, .4)
save_image(im, "overflow")

# 5. Clamp Image
clamp_image(im)
save_image(im, "doglight_fixed")
'''

# Colorspace and saturation
# Saturates the swatches image
im = load_image("data/swatch.jpg")
rgb_to_hsv(im)
# can change the third parameter to increase/decrease saturation below
# decimals r ok
shift_image(im, 1, 1)
clamp_image(im)
hsv_to_rgb(im)
save_image(im, "swatch_saturated")

# Black and White filter on skin swatch
im = load_image("data/swatch.jpg")
im = rgb_to_grayscale(im)
save_image(im, "1swatch")


