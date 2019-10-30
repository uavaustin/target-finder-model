from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
import os
import createUnrotatedConfig 
import random

def _get_base_shapes(shape):
    """Get the base shape images for a given shapes"""
    # For now just using the first one to prevent bad alpha placement
    # TODO: Use more base shapes
    base_path = os.path.join(createUnrotatedConfig.BASE_SHAPES_DIR,
                             shape,
                             '{}-01.png'.format(shape))
    return Image.open(base_path)


def _augment_color(color_rgb):
    """Shift the color a bit"""
    (r, g, b) = color_rgb
    r = max(min(r + random.randint(-10, 11), 255), 1)
    g = max(min(g + random.randint(-10, 11), 255), 1)
   
    return (r, g, b)

def _get_base(base, target_rgb):
    """Copy and recolor the base shape"""
    image = base.copy()
    image = image.resize((256, 256), 0)
    image = image.convert('RGBA')
    (r, g, b) = target_rgb

    for x in range(image.width):
        for y in range(image.height):

            pr, pg, pb, _ = image.getpixel((x, y))

            if pr != 255 or pg != 255 or pb != 255:
                image.putpixel((x, y), (r, g, b, 255))

    return image

def _strip_image(image):
    """Remove white and black edges"""
    for x in range(image.width):
        for y in range(image.height):

            r, g, b, a = image.getpixel((x, y))

            if r == 255 and g == 255 and b == 255:
                image.putpixel((x, y), (0, 0, 0, 0))

    image = image.crop(image.getbbox())

    return image

def _random_list(items, count):
    """Get a list of items with length count"""
    return [random.choice(items) for i in range(0, count)]



def _add_alphanumeric(image, shape, alpha, alpha_rgb, font_file):
    # Adjust alphanumeric size based on the shape it will be on
    if shape == 'star':
        font_multiplier = 0.14
    if shape == 'triangle':
        font_multiplier = 0.5
    elif shape == 'rectangle':
        font_multiplier = 0.72
    elif shape == 'quarter-circle':
        font_multiplier = 0.60
    elif shape == 'semicircle':
        font_multiplier = 0.55
    elif shape == 'circle':
        font_multiplier = 0.55
    elif shape == 'square':
        font_multiplier = 0.60
    elif shape == 'trapezoid':
        font_multiplier = 0.60
    else:
        font_multiplier = 0.55

    # Set font size, select font style from fonts file, set font color
    font_size = int(round(font_multiplier * image.height))
    font = ImageFont.truetype(font_file, font_size)
    draw = ImageDraw.Draw(image)

    w, h = draw.textsize(alpha, font=font)

    x = (image.width - w) / 2
    y = (image.height - h) / 2

    # Adjust centering of alphanumerics on shapes
    if shape == 'pentagon':
        pass
    elif shape == 'semicircle':
        pass
    elif shape == 'rectangle':
        pass
    elif shape == 'trapezoid':
        y -= 20
    elif shape == 'star':
        pass
    elif shape == 'triangle':
        x -= 24
        y += 12
    elif shape == 'quarter-circle':
        y -= 40
        x += 14
    elif shape == 'cross':
        y -= 25
    elif shape == 'square':
        y -= 10
    elif shape == 'circle':
        pass
    else:
        pass

    draw.text((x, y), alpha, alpha_rgb, font=font)

    return image
