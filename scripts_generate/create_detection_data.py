#!/usr/bin/env python3
"""
This script should generate fullsized training images
which contain several artificial shapes.

Save Format:
    data/{train, val}/images/exX.png <- image X
    data/{train, val}/images/exX.txt <- bboxes for image X

BBox Format:
    shape_ALPHA x y width height
    shape2_ALPHA2 x y width height
    ...
"""
import glob
import multiprocessing
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageFile

from tqdm import tqdm

import generate_config as config


# Get constants from config
NUM_GEN = int(config.NUM_IMAGES)
MAX_SHAPES = int(config.MAX_PER_SHAPE)
FULL_SIZE = config.FULL_SIZE
TARGET_COLORS = config.TARGET_COLORS
ALPHA_COLORS = config.ALPHA_COLORS
COLORS = config.COLORS
CLASSES = config.OD_CLASSES
ALPHAS = config.ALPHAS


def generate_all_images(gen_type, num_gen, offset=0):
    """Generate the full sized images"""
    images_dir = os.path.join(config.DATA_DIR, gen_type, "images")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    r_state = random.getstate()
    random.seed(gen_type + str(offset))

    # All the random selection is generated ahead of time, that way
    # the process can be resumed without the shapes changing on each
    # run.

    base_shapes = {}
    for shape in config.SHAPE_TYPES:
        base_shapes[shape] = _get_base_shapes(shape)

    numbers = list(range(offset, num_gen + offset))

    backgrounds = _random_list(_get_backgrounds(), num_gen)
    flip_bg = _random_list([False, True], num_gen)
    mirror_bg = _random_list([False, True], num_gen)
    sharpen = _random_list(range(0, 4), num_gen)
    blurs = _random_list(range(1, 3), num_gen)
    num_targets = _random_list(range(1, MAX_SHAPES), num_gen)

    crop_xs = _random_list(range(0, config.FULL_SIZE[0] - config.CROP_SIZE[0]), num_gen)
    crop_ys = _random_list(range(0, config.FULL_SIZE[1] - config.CROP_SIZE[1]), num_gen)

    num_targets = 1
    shape_params = []

    for i in range(num_gen):
        shape_names = _random_list(config.SHAPE_TYPES, num_targets)
        bases = [random.choice(base_shapes[shape]) for shape in shape_names]
        alphas = _random_list(config.ALPHAS, num_targets)
        font_files = _random_list(config.ALPHA_FONTS, num_targets)

        target_colors = _random_list(TARGET_COLORS, num_targets)
        alpha_colors = _random_list(ALPHA_COLORS, num_targets)

        # Make sure shape and alpha are different colors
        for i, target_color in enumerate(target_colors):
            while alpha_colors[i] == target_color:
                alpha_colors[i] = random.choice(ALPHA_COLORS)

        target_rgbs = [random.choice(COLORS[color]) for color in target_colors]
        alpha_rgbs = [random.choice(COLORS[color]) for color in alpha_colors]

        sizes = _random_list(range(40, 65), num_targets)

        angles = _random_list(range(0, 360), num_targets)

        xs = _random_list(range(70, config.CROP_SIZE[0] - 70, 20), num_targets)
        ys = _random_list(range(70, config.CROP_SIZE[1] - 70, 20), num_targets)

        shape_params.append(
            list(
                zip(
                    shape_names,
                    bases,
                    alphas,
                    font_files,
                    sizes,
                    angles,
                    target_colors,
                    target_rgbs,
                    alpha_colors,
                    alpha_rgbs,
                    xs,
                    ys,
                )
            )
        )

    # Put everything into one large iterable so that we can split up
    # data across thread pools.ImageFile.LOAD_TRUNCATED_IMAGES = True
    data = zip(
        numbers,
        backgrounds,
        crop_xs,
        crop_ys,
        flip_bg,
        mirror_bg,
        sharpen,
        blurs,
        shape_params,
        [gen_type] * num_gen,
    )

    random.setstate(r_state)

    # Generate in a pool. If specificed, use a given number of
    # threads.
    with multiprocessing.Pool(None) as pool:
        processes = pool.imap_unordered(_generate_single_example, data)
        for i in tqdm(processes, total=num_gen):
            pass


def _generate_single_example(data):
    """Creates a single full image"""
    (
        number,
        background,
        crop_x,
        crop_y,
        flip_bg,
        mirror_bg,
        sharpen,
        blur,
        shape_params,
        gen_type,
    ) = data

    data_path = os.path.join(config.DATA_DIR, gen_type, "images")
    labels_fn = os.path.join(data_path, "ex{}.txt".format(number))
    img_fn = os.path.join(data_path, "ex{}.{}".format(number, config.IMAGE_EXT))

    background = background.copy()
    background = background.crop(
        (crop_x, crop_y, crop_x + config.CROP_SIZE[0], crop_y + config.CROP_SIZE[1])
    )

    if flip_bg:
        background = ImageOps.flip(background)
    if mirror_bg:
        background = ImageOps.mirror(background)

    shape_imgs = [_create_shape(*shape_param) for shape_param in shape_params]

    shape_bboxes, full_img = _add_shapes(background, shape_imgs, shape_params, blur)

    if sharpen == 1:
        full_img = full_img.filter(ImageFilter.SHARPEN)

    full_img = full_img.resize(config.DETECTOR_SIZE)
    full_img.save(img_fn)

    with open(labels_fn, "w") as label_file:
        for shape_bbox in shape_bboxes:
            label_file.write("{} {} {} {} {}\n".format(*shape_bbox))


def _add_shapes(background, shape_imgs, shape_params, blur_radius):
    """Paste shapes onto background and return bboxes"""
    shape_bboxes = []

    for i, shape_param in enumerate(shape_params):

        x = shape_param[-2]
        y = shape_param[-1]
        shape_img = shape_imgs[i]

        x1, y1, x2, y2 = shape_img.getbbox()
        bg_at_shape = background.crop((x1 + x, y1 + y, x2 + x, y2 + y))
        bg_at_shape.paste(shape_img, (0, 0), shape_img)
        bg_at_shape = bg_at_shape.filter(ImageFilter.GaussianBlur(blur_radius))
        background.paste(bg_at_shape, (x, y))

        im_w, im_h = background.size
        x /= im_w
        y /= im_h

        w = (x2 - x1) / im_w
        h = (y2 - y1) / im_h

        shape_bboxes.append((CLASSES.index(shape_param[0]), x, y, w, h))
        shape_bboxes.append((CLASSES.index(shape_param[2]), x, y, w, h))

    return shape_bboxes, background.convert("RGB")


def _get_backgrounds():
    """Get the background assets"""
    # Can be a mix of .png and .jpg
    filenames = glob.glob(os.path.join(config.BACKGROUNDS_DIR, "*.png"))
    filenames += glob.glob(os.path.join(config.BACKGROUNDS_DIR, "*.jpg"))

    return [Image.open(img).resize(config.FULL_SIZE) for img in filenames]


def _get_base_shapes(shape):
    """Get the base shape images for a given shapes"""
    # For now just using the first one to prevent bad alpha placement
    base_path = os.path.join(config.BASE_SHAPES_DIR, shape, "{}-01.png".format(shape))
    return [Image.open(base_path)]


def _random_list(items, count):
    """Get a list of items with length count"""
    return [random.choice(items) for i in range(0, count)]


def _create_shape(
    shape,
    base,
    alpha,
    font_file,
    size,
    angle,
    target_color,
    target_rgb,
    alpha_color,
    alpha_rgb,
    x,
    y,
):
    """Create a shape given all the input parameters"""
    target_rgb = _augment_color(target_rgb)
    alpha_rgb = _augment_color(alpha_rgb)

    image = _get_base(base, target_rgb, size)
    image = _strip_image(image)
    image = _add_alphanumeric(image, shape, alpha, alpha_rgb, font_file)

    w, h = image.size
    ratio = min(size / w, size / h)
    image = image.resize((int(w * ratio), int(h * ratio)), 1)

    image = _rotate_shape(image, shape, angle)
    image = _strip_image(image)

    return image


def _augment_color(color_rgb):
    """Shift the color a bit"""
    r, g, b = color_rgb
    r = max(min(r + random.randint(-10, 11), 255), 1)
    g = max(min(g + random.randint(-10, 11), 255), 1)
    b = max(min(b + random.randint(-10, 11), 255), 1)
    return (r, g, b)


def _get_base(base, target_rgb, size):
    """Copy and recolor the base shape"""
    image = base.copy()
    image = image.resize((256, 256), 1)
    image = image.convert("RGBA")

    r, g, b = target_rgb

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


def _add_alphanumeric(image, shape, alpha, alpha_rgb, font_file):
    # Adjust alphanumeric size based on the shape it will be on
    if shape == "star":
        font_multiplier = 0.14
    if shape == "triangle":
        font_multiplier = 0.5
    elif shape == "rectangle":
        font_multiplier = 0.72
    elif shape == "quarter-circle":
        font_multiplier = 0.60
    elif shape == "semicircle":
        font_multiplier = 0.55
    elif shape == "circle":
        font_multiplier = 0.55
    elif shape == "square":
        font_multiplier = 0.60
    elif shape == "trapezoid":
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
    if shape == "pentagon":
        pass
    elif shape == "semicircle":
        pass
    elif shape == "rectangle":
        pass
    elif shape == "trapezoid":
        y -= 20
    elif shape == "star":
        pass
    elif shape == "triangle":
        x -= 24
        y += 12
    elif shape == "quarter-circle":
        y -= 40
        x += 14
    elif shape == "cross":
        y -= 25
    elif shape == "square":
        y -= 10
    elif shape == "circle":
        pass
    else:
        pass

    draw.text((x, y), alpha, alpha_rgb, font=font)

    return image


def _rotate_shape(image, shape, angle):
    return image.rotate(angle, expand=1)


if __name__ == "__main__":

    generate_all_images("detector_train", config.NUM_IMAGES, config.NUM_OFFSET)

    generate_all_images("detector_val", config.NUM_VAL_IMAGES, config.NUM_VAL_OFFSET)
