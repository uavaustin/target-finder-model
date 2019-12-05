#!/usr/bin/env python3

from tqdm import tqdm
from PIL import Image, ImageFilter, ImageOps
import multiprocessing
import random
import generate_config as config
import glob
import os

from create_detection_data import _random_list, _get_backgrounds


# Get constants from config
CLF_WIDTH, CLF_HEIGHT = config.PRECLF_SIZE
CROP_WIDTH, CROP_HEIGHT = config.CROP_SIZE
FILE_PATH = os.path.abspath(os.path.dirname(__file__))

CLASSES = config.SHAPE_TYPES + config.ALPHAS

def create_det_images():
    """Generate data for the classifier model"""

    imgs = glob.glob('scripts_generate/data/detector_train_270/images/*.png')
    txts = glob.glob('scripts_generate/data/detector_train_270/images/*.txt')

    save_path = 'scripts_generate/data/detector_train/images'

    os.makedirs(save_path, exist_ok=True)
    
    data = zip(imgs, txts, [save_path]*len(imgs))

    with multiprocessing.Pool(None) as pool:
        processes = pool.imap_unordered(convert, data)
        for i in tqdm(processes, total=len(imgs)):
            pass


    imgs = glob.glob('scripts_generate/data/detector_val_270/images/*.png')
    txts = glob.glob('scripts_generate/data/detector_val_270/images/*.txt')

    save_path = 'scripts_generate/data/detector_val/images'

    os.makedirs(save_path, exist_ok=True)

    data = zip(imgs, txts, [save_path]*len(imgs))

    with multiprocessing.Pool(None) as pool:
        processes = pool.imap_unordered(convert, data)
        for i in tqdm(processes, total=len(imgs)):
            pass


def convert(data):
    img, txt, save_path = data
    im = Image.open(img)
    img_name = os.path.basename(img)
    sharp = random.randint(1,8)
    if sharp == 1:
        im = im.filter(ImageFilter.SHARPEN)
    im.save(os.path.join(save_path, img_name))


    with open(txt, 'r') as label_file:
        for line in label_file.readlines():
            class_desc, x, y, w, h = line.strip().split(' ')
            x, y, w, h = float(x), float(y), float(w), float(h)
    
    txt_name = os.path.basename(txt)
    save_txt = os.path.join(save_path, txt_name)
    shape_alpha = txt_name.split('_')[0]
    shape = shape_alpha.split('-')[0]
    alpha = shape_alpha.split('-')[1]

    if "circle" in alpha:
        shape = "quarter-circle"
        alpha = shape_alpha.split('e-')[1]

    with open(save_txt, 'w') as new_txt:
        new_txt.write('{} {} {} {} {}\n'.format(CLASSES.index(shape), x, y, w, h))
        new_txt.write('{} {} {} {} {}\n'.format(CLASSES.index(alpha), 1.1*x, 1.1*y, 0.9*w, 0.9*h))


    return

if __name__ == "__main__":

    create_det_images()
