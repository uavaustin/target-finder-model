#!/usr/bin/env python3

import glob
import multiprocessing
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from tqdm import tqdm

import generate_config as config
import create_detection_data as gen_lib

import generate_config as config

TARGET_COLORS = config.TARGET_COLORS
ALPHA_COLORS = config.ALPHA_COLORS
COLORS = config.COLORS


def create_orientation_data(gen_type, num_gen, offset=0):
    """Create orientation data"""
    images_dir = os.path.join(config.DATA_DIR, gen_type, 'images')
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Seed the data gen
    r_state = random.getstate()
    random.seed(gen_type + str(offset))

    # Get base shapes
    base_shapes = {}
    for shape in config.SHAPE_TYPES:
        base_shapes[shape] = gen_lib._get_base_shapes(shape)

    shape_params = []

    numbers = list(range(offset, offset + num_gen))
    backgrounds = gen_lib._random_list(_get_backgrounds((65, 65)), num_gen)
    flip_bg = gen_lib._random_list([False, True], num_gen)
    mirror_bg = gen_lib._random_list([False, True], num_gen)
    blurs = gen_lib._random_list(range(1, 3), num_gen)

    for i in range(num_gen):
        angles = gen_lib._random_list(range(0, 360, 90), 1)
        alphas = gen_lib._random_list(config.ALPHAS, 1)
        font_files = gen_lib._random_list(config.ALPHA_FONTS, 1)
        target_colors = gen_lib._random_list(TARGET_COLORS, 1)
        alpha_colors = gen_lib._random_list(ALPHA_COLORS, 1)
        target_rgbs = [random.choice(COLORS[color]) for color in target_colors]
        alpha_rgbs = [random.choice(COLORS[color]) for color in alpha_colors]
        sizes = gen_lib._random_list(range(35, 40), 1)
        shape_names = gen_lib._random_list(config.SHAPE_TYPES, 1)
        bases = [random.choice(base_shapes[shape_names[0]])]

        xs = gen_lib._random_list(range(5, 10, 2), 1)
        ys = gen_lib._random_list(range(5, 10, 2), 1)

    shape_params.append(list(zip(shape_names, bases, alphas,
                                 font_files, sizes, angles,
                                 target_colors, target_rgbs,
                                 alpha_colors, alpha_rgbs,
                                 xs, ys)))

    data = zip(numbers, backgrounds, flip_bg, mirror_bg, [gen_type] * num_gen, shape_params, blurs)

    with multiprocessing.Pool(None) as pool:
        processes = pool.imap_unordered(_generate_single_orientation, data)
        for i in tqdm(processes, total=num_gen):
            pass


def _generate_single_orientation(data):

    number, background, flip_bg, mirror_bg, gen_type, shape_params, blur = data

    alpha = shape_params[0][2]
    angle = shape_params[0][5]

    background = background.copy()
    if flip_bg:
        background = ImageOps.flip(background)
    if mirror_bg:
        background = ImageOps.mirror(background)

    shape_imgs = [(_create_shape(*shape_params[0])).resize((55, 55))]
    shape_bboxes, full_img = gen_lib._add_shapes(background, shape_imgs,
                                                 shape_params, blur)

    save_path = os.path.join(config.DATA_DIR, gen_type, 'images')
    if angle != 0:
        angle = 360 - angle

    img_fn = os.path.join(save_path, 'ex{}-{}.{}'.format(alpha, angle, config.IMAGE_EXT))

    full_img.save(img_fn)


def _create_shape(shape, base, alpha,
                  font_file, size, angle,
                  target_color, target_rgb,
                  alpha_color, alpha_rgb, x, y):
    """Create a shape given all the input parameters"""
    target_rgb = gen_lib._augment_color(target_rgb)
    alpha_rgb = gen_lib._augment_color(alpha_rgb)

    image = gen_lib._get_base(base, target_rgb, size)
    image = gen_lib._strip_image(image)
    image = gen_lib._add_alphanumeric(image, shape, alpha, alpha_rgb, font_file)

    w, h = image.size
    ratio = min(size / w, size / h)
    image = image.resize((int(w * ratio), int(h * ratio)), 1)

    image = gen_lib._rotate_shape(image, shape, angle)
    image = gen_lib._strip_image(image)

    return image


def _get_backgrounds(SIZE):
    """Get the background assets"""
    # Can be a mix of .png and .jpg
    filenames = glob.glob(os.path.join(config.BACKGROUNDS_DIR, '*.png'))
    filenames += glob.glob(os.path.join(config.BACKGROUNDS_DIR, '*.jpg'))

    x = random.randint(0, config.FULL_SIZE[0] - SIZE[0])
    y = random.randint(0, config.FULL_SIZE[1] - SIZE[1])

    return [(Image.open(filename).resize(config.FULL_SIZE)).crop((x, y, x + SIZE[0], y + SIZE[1]))
            for filename in sorted(filenames)]


if __name__ == "__main__":
    if config.NUM_IMAGES != 0:
        create_orientation_data('orient_train', config.NUM_IMAGES, config.NUM_OFFSET)

    if config.NUM_VAL_IMAGES != 0:
        create_orientation_data('orient_val', config.NUM_VAL_IMAGES, config.NUM_VAL_OFFSET)
