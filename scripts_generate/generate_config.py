"""
Contains configuration settings for generation.
"""

import pathlib
import yaml

with pathlib.Path("config.yaml").open("r") as stream:
    config = yaml.safe_load(stream)


# [Asset Settings and Files]
BACKGROUNDS_VERSION = config["generate"]["backgrounds_version"]
BASE_SHAPES_VERSION = config["generate"]["base_shapes_version"]

DOWNLOAD_BASE = config["generate"]["download_base_url"]

BACKGROUNDS_URL = f"{DOWNLOAD_BASE}backgrounds-{BACKGROUNDS_VERSION}.tar.gz"
BASE_SHAPES_URL = f"{DOWNLOAD_BASE}base-shapes-{BASE_SHAPES_VERSION}.tar.gz"

ASSETS_DIR = pathlib.Path(__file__).parent / "assets"

BACKGROUNDS_DIR = ASSETS_DIR / f"backgrounds-{BACKGROUNDS_VERSION}"
BASE_SHAPES_DIR = ASSETS_DIR / f"base-shapes-{BASE_SHAPES_VERSION}"

DATA_DIR = pathlib.Path(__file__).parent / "data"

# [Number of Images]
# Generate num - offset images
NUM_OFFSET = config["generate"]["train_batch"]["offset"]
NUM_IMAGES = config["generate"]["train_batch"]["images"]
NUM_VAL_OFFSET = config["generate"]["eval_batch"]["offset"]
NUM_VAL_IMAGES = config["generate"]["eval_batch"]["images"]

# Max images to generate per image
MAX_PER_SHAPE = config["generate"]["max_shapes_per_image"]

# Specify number of threads to use for shape generation. Default lets
# the multiprocessing library determine.
NUM_THREADS = config["generate"]["threads"]

# [Shape Specs]
SHAPE_TYPES = config["classes"]["shapes"]

CLF_TYPES = ["background", "target"]

TARGET_COLORS = [
    "white",
    "black",
    "gray",
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
]

ALPHA_COLORS = [
    "white",
    "black",
    "gray",
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
]

COLORS = {
    "white": [(240, 239, 233)],
    "black": [(5, 5, 5)],
    "gray": [(128, 128, 128)],
    "red": [(188, 60, 60), (255, 80, 80), (255, 0, 0), (154, 0, 0)],
    "blue": [(0, 0, 255), (0, 0, 135)],
    "green": [(64, 115, 64), (148, 255, 148), (0, 255, 0), (0, 128, 4)],
    "yellow": [(225, 221, 104), (255, 252, 122), (255, 247, 0), (210, 203, 0)],
    "purple": [(127, 127, 255), (128, 0, 128)],
    "orange": [
        (153, 76, 0),
        (216, 172, 83),
        (255, 204, 101),
        (255, 165, 0),
        (210, 140, 0),
    ],
}

ALPHAS = config["classes"]["alphas"]

ALPHA_FONT_DIR = pathlib.Path(__file__).parent / "vendor" / "fonts"
ALPHA_FONTS = [
    ALPHA_FONT_DIR / "Rajdhani" / "Rajdhani-Bold.ttf",
    ALPHA_FONT_DIR / "Gudea" / "Gudea-Bold.ttf",
    ALPHA_FONT_DIR / "Inconsolata" / "Inconsolata-Bold.ttf",
    ALPHA_FONT_DIR / "Open_Sans" / "OpenSans-Bold.ttf",
    ALPHA_FONT_DIR / "Open_Sans" / "OpenSans-SemiBold.ttf",
    ALPHA_FONT_DIR / "News_Cycle" / "NewsCycle-Bold.ttf",
]

OD_CLASSES = SHAPE_TYPES + ALPHAS

# [Model Dimensions]
FULL_SIZE = (
    config["inputs"]["full_image"]["width"],
    config["inputs"]["full_image"]["height"],
)
CROP_SIZE = (
    config["inputs"]["cropping"]["width"],
    config["inputs"]["cropping"]["height"],
)
CROP_OVERLAP = config["inputs"]["cropping"]["overlap"]
DETECTOR_SIZE = (
    config["inputs"]["detector"]["width"],
    config["inputs"]["detector"]["height"],
)
PRECLF_SIZE = (
    config["inputs"]["preclf"]["width"],
    config["inputs"]["preclf"]["height"],
)

# Whether to delete full image data when they are converted
DELETE_ON_CONVERT = config["generate"]["delete_on_convert"]

IMAGE_EXT = config["generate"]["img_ext"]
