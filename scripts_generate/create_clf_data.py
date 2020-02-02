#!/usr/bin/env python3

from tqdm import tqdm
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import multiprocessing
import random
import generate_config as config
import glob
import os
import numpy as np

from create_detection_data import _random_list, _get_backgrounds

# Get constants from config
CLF_WIDTH, CLF_HEIGHT = config.PRECLF_SIZE
CROP_WIDTH, CROP_HEIGHT = config.CROP_SIZE
FILE_PATH = os.path.abspath(os.path.dirname(__file__))


def create_clf_images(gen_type, num_gen, offset=0):
    """Generate data for the classifier model"""

    save_dir = os.path.join(config.DATA_DIR, gen_type, "images")
    os.makedirs(save_dir, exist_ok=True)

    # Get target images
    data_folder = "detector_" + gen_type.split("_")[1]
    images_dir = os.path.join(
        config.DATA_DIR, data_folder, "images/*" + str(config.IMAGE_EXT)
    )
    image_names = glob.glob(images_dir)
    image_names = _random_list(image_names, num_gen)

    numbers = list(range(offset, offset + num_gen))

    # Get random crops and augmentations for background
    backgrounds = _random_list(_get_backgrounds(), num_gen)
    flip_bg = _random_list([False, True], num_gen)
    mirror_bg = _random_list([False, True], num_gen)
    blurs = _random_list(range(1, 3), num_gen)
    enhancements = _random_list(np.linspace(0.5, 2, 5), num_gen)
    crop_xs = _random_list(range(0, config.FULL_SIZE[0] - config.CROP_SIZE[0]), num_gen)
    crop_ys = _random_list(range(0, config.FULL_SIZE[1] - config.CROP_SIZE[1]), num_gen)

    gen_types = [gen_type] * num_gen

    data = zip(
        numbers,
        backgrounds,
        crop_xs,
        crop_ys,
        flip_bg,
        mirror_bg,
        blurs,
        enhancements,
        image_names,
        gen_types,
    )

    with multiprocessing.Pool(None) as pool:
        processes = pool.imap_unordered(_single_clf_image, data)
        for i in tqdm(processes, total=num_gen):
            pass


def _single_clf_image(data):
    """Crop detection image and augment clf image and save"""
    (
        number,
        background,
        crop_x,
        crop_y,
        flip_bg,
        mirror_bg,
        blur,
        enhancement,
        shape_img,
        gen_type,
    ) = data

    background = background.copy()
    background = background.crop(
        (crop_x, crop_y, crop_x + config.CROP_SIZE[0], crop_y + config.CROP_SIZE[1])
    )

    if flip_bg:
        background = ImageOps.flip(background)
    if mirror_bg:
        background = ImageOps.mirror(background)

    background.filter(ImageFilter.GaussianBlur(blur))
    background = background.resize(config.PRECLF_SIZE)
    background = _enhance_image(background, enhancement)

    data_path = os.path.join(config.DATA_DIR, gen_type, "images")
    bkg_fn = os.path.join(
        data_path, "background_{}.{}".format(number, config.IMAGE_EXT)
    )
    background.save(bkg_fn)

    # Now consider the shape image
    shape = Image.open(shape_img).resize(config.PRECLF_SIZE)
    shape = _enhance_image(shape, enhancement)
    shape_fn = os.path.join(data_path, "target_{}.{}".format(number, config.IMAGE_EXT))
    shape.save(shape_fn)


def _enhance_image(img, enhancement):
    converter = ImageEnhance.Color(img)
    return converter.enhance(enhancement)


if __name__ == "__main__":

    if config.NUM_IMAGES != 0:
        create_clf_images("clf_train", config.NUM_IMAGES, config.NUM_OFFSET)

    if config.NUM_VAL_IMAGES != 0:
        create_clf_images("clf_val", config.NUM_VAL_IMAGES, config.NUM_VAL_OFFSET)
