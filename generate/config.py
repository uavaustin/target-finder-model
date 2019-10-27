"""Contains configuration settings for generation."""

import os
import random


# [Asset Settings and Files]
BACKGROUNDS_VERSION = 'v2'
BASE_SHAPES_VERSION = 'v1'

DOWNLOAD_BASE = (
    'https://bintray.com/uavaustin/target-finder-assets/'
    'download_file?file_path='
)

BACKGROUNDS_URL = (
    DOWNLOAD_BASE + 'backgrounds-' + BACKGROUNDS_VERSION + '.tar.gz'
)
BASE_SHAPES_URL = (
    DOWNLOAD_BASE + 'base-shapes-' + BASE_SHAPES_VERSION + '.tar.gz'
)

ASSETS_DIR = os.environ.get('ASSETS_DIR',
                            os.path.join(os.path.dirname(__file__), 'assets'))

BACKGROUNDS_DIR = os.path.join(ASSETS_DIR,
                               'backgrounds-' + BACKGROUNDS_VERSION)
BASE_SHAPES_DIR = os.path.join(ASSETS_DIR,
                               'base-shapes-' + BASE_SHAPES_VERSION)


DATA_DIR = os.environ.get('DATA_DIR',
                          os.path.join(os.path.dirname(__file__), 'data'))

# [Number of Images]
# Generate num - offset images
NUM_OFFSET = int(os.environ.get('NUM_OFFSET', '0'))
NUM_IMAGES = int(os.environ.get('NUM_IMAGES', '100'))
NUM_VAL_OFFSET = int(os.environ.get('NUM_VAL_OFFSET', '0'))
NUM_VAL_IMAGES = int(os.environ.get('NUM_VAL_IMAGES', '10'))

# Max images to generate per image
MAX_PER_SHAPE = int(os.environ.get('MAX_PER_SHAPE', '2'))

# Specify number of threads to use for shape generation. Default lets
# the multiprocessing library determine.
NUM_THREADS = int(os.environ.get('NUM_THREADS', 0))

# [Shape Specs]
SHAPE_TYPES = os.environ.get(
    'SHAPE_TYPES',
    'circle,cross,pentagon,quarter-circle,rectangle,semicircle,square,star,'
    'trapezoid,triangle'
).split(',')

CLF_TYPES = ['background', 'shape_target']

TARGET_COLORS = ['white', 'black','gray', 'red', 'blue',
                 'green', 'yellow', 'purple', 'orange', 'brown']

ALPHA_COLORS = ['white', 'black', 'gray', 'red', 'blue',
                'green', 'yellow', 'purple', 'orange', 'brown']

def red():
    if(random.randint(0,1) == 3):
        r = random.randint(220,255)
        g = random.randint(0,180)
        b = g + random.randint(-26,25)
        if(b < 0):
            b = 0   
        return (r,g,b)
    else:
        r = random.randint(175,219)
        g = (int)(r/2) + random.randint(-10,10)
        b = g + random.randint(-10,10)
        return (r,g,b)


def blue():
    r = random.randint(0,100)
    g = random.randint(80,145)
    b = random.randint(180,255)
    return (r,g,b)

def green():
    g = random.randint(130,255)
    r = g - random.randint(40,100)
    b = r + random.randint(-30,30)
    return (r,g,b)

def purple():
    r = random.randint(100,160)
    g = r - random.randint(50,100)
    b = r + random.randint(30,80)
    return (r,g,b)

def orange():
    r = random.randint(235,255)
    g = random.randint(130,180)
    b = g - random.randint(70,120)
    return (r,g,b)

def yellow():
    r = random.randint(235,255)
    g = random.randint(220,240)
    b = g - random.randint(80,240)
    if(b<0):
        b = 0
    return (r,g,b)

def white():
    r = random.randint(245,248)
    g = r + random.randint(245,248)
    b = r + random.randint(245,248)
    return (r,g,b)

def black():
    r = random.randint(20,75)
    g = r + random.randint(-6,5)
    b = r + random.randint(10,30)
    return (r,g,b)

def brown():
    if(random.randint(0,1) == 0):
        r = random.randint(90,110)
        g = r - random.randint(30,50)
        b = g - random.randint(0,10)
        return (r,g,b)
    else:
        r = random.randint(190,220)
        g = r - random.randint(30,50)
        b = g - random.randint(30,50)
        return (r,g,b)

def gray():
    r = random.randint(150,215)
    g = r + random.randint(-6,5)
    b = r + random.randint(-6,5)
    return (r,g,b)

COLORS = {
    'white': [white()],
    'black': [black()],
    'gray': [gray()],
    'red': [red()],
    'blue': [blue()],
    'green': [green()],
    'yellow': [yellow()],
    'purple': [purple()],
    'orange': [orange()],
    'brown': [brown()]
}

ALPHAS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ4')

ALPHA_FONT_DIR = os.path.join(os.path.dirname(__file__), 'vendor', 'fonts')
ALPHA_FONTS = [
    os.path.join(ALPHA_FONT_DIR, 'Rajdhani', 'Rajdhani-Bold.ttf'),
    os.path.join(ALPHA_FONT_DIR, 'Gudea', 'Gudea-Bold.ttf'),
    os.path.join(ALPHA_FONT_DIR, 'Inconsolata', 'Inconsolata-Bold.ttf'),
    os.path.join(ALPHA_FONT_DIR, 'Open_Sans', 'OpenSans-Bold.ttf'),
    os.path.join(ALPHA_FONT_DIR, 'Open_Sans', 'OpenSans-SemiBold.ttf'),
    os.path.join(ALPHA_FONT_DIR, 'News_Cycle', 'NewsCycle-Bold.ttf')
]

YOLO_CLASSES = SHAPE_TYPES + ALPHAS

# [Model Dimensions]
FULL_SIZE = (4240, 2400)
CROP_SIZE = (400, 400)
CROP_OVERLAP = 100
DETECTOR_SIZE = (608, 608)
PRECLF_SIZE = (64, 64)

# [Darknet Models]

# Whether to delete full image data when they are converted
DELETE_ON_CONVERT = False
