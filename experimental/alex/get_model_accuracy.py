#!/usr/bin/env python3
"""A simple script to visualize model performance on an image.
Usage: PYTHONPATH=$(pwd) experimental/alex/get_model_accuracy.py
"""

import pathlib
import time
from typing import List, Tuple, Dict

from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import numpy as np

from target_finder_model import inference

with pathlib.Path("config.yaml").open("r") as stream:
    import yaml

    CONFIG = yaml.safe_load(stream)


def draw_image(img_path: pathlib.Path, clf_results, det_results, crops):
    """Crop the input image."""
    croppage = CONFIG["inputs"]["cropping"]

    img = Image.open(img_path.open("rb"))
    img = img.convert("RGBA")
    w, h = img.size

    TINT_COLOR = (0, 0, 0)  # Black
    TRANSPARENCY = 0.25  # Degree of transparency, 0-100%
    OPACITY = int(255 * TRANSPARENCY)

    slice_num = 0
    for y1 in range(0, h, croppage["height"] - croppage["overlap"]):
        for x1 in range(0, w, croppage["width"] - croppage["overlap"]):
            if clf_results[slice_num].class_idx == 1:
                if y1 + croppage["height"] > h:
                    y1 = h - croppage["height"]

                if x1 + croppage["width"] > w:
                    x1 = w - croppage["width"]

                y2 = y1 + croppage["height"]
                x2 = x1 + croppage["width"]

                overlay = Image.new("RGBA", img.size, TINT_COLOR + (0,))
                draw = ImageDraw.Draw(overlay)
                draw.rectangle(((x1, y1), (x2, y2)), fill=TINT_COLOR + (OPACITY,))

                img = Image.alpha_composite(img, overlay)

            slice_num += 1

    draw = ImageDraw.Draw(img)
    for tile_dets, region in zip(det_results, crops):

        ratio_x = CONFIG["inputs"]["detector"]["width"] / (region[2] - region[0])
        ratio_y = CONFIG["inputs"]["detector"]["height"] / (region[3] - region[1])

        for det in tile_dets:
            if det.confidence < 0.7:
                continue
            det.x /= ratio_x
            det.width /= ratio_x
            det.y /= ratio_x
            det.height /= ratio_y

            draw.rectangle(
                [
                    det.x + region[0],
                    det.y + region[1],
                    det.x + det.width + region[0],
                    det.y + det.height + region[1],
                ],
                outline="green",
                width=3,
            )
            draw.text((det.x + region[0], det.y + region[1] - 10), det.class_name)

    img = img.resize((int(w / 2), int(h / 2)))
    img.show()


def process_image(img_path: pathlib.Path) -> List[Image.Image]:
    """Crop the input image."""
    croppage = CONFIG["inputs"]["cropping"]
    clf = CONFIG["inputs"]["preclf"]

    img = Image.open(img_path.open("rb"))
    w, h = img.size

    crops, regions = [], []

    for y1 in range(0, h, croppage["height"] - croppage["overlap"]):
        for x1 in range(0, w, croppage["width"] - croppage["overlap"]):

            if y1 + croppage["height"] > h:
                y1 = h - croppage["height"]

            if x1 + croppage["width"] > w:
                x1 = w - croppage["width"]

            y2 = y1 + croppage["height"]
            x2 = x1 + croppage["width"]

            # Reisze the image for CLF
            box = img.crop((x1, y1, x2, y2)).resize((clf["width"], clf["height"]))
            crops.append(box)
            regions.append((x1, y1, x2, y2))

    return crops, regions


def resize_crops(crops: List[Image.Image], size: Dict[str, int]) -> List[Image.Image]:
    return np.array(
        [np.array(crop.resize((size["width"], size["height"]))) for crop in crops]
    )


if __name__ == "__main__":

    img_path = pathlib.Path("scripts_generate/data/fixtures/real-2.jpg")

    models = {
        "frcnn": inference.DetectionModel(),
        "clf": inference.ClfModel("target_finder_model/data/clf"),
    }

    models["frcnn"].load()
    models["clf"].load()

    detector = models["frcnn"]
    clf = models["clf"]

    start = time.perf_counter()
    # Get the image crops
    crops, regions = process_image(img_path)
    # Resize to classifier size
    clf_crops = resize_crops(crops, CONFIG["inputs"]["preclf"])
    clf_res = clf.predict(clf_crops)

    target_crops = [
        (crops[i], regions[i])
        for i, region in enumerate(clf_res)
        if region.class_idx == 1
    ]

    image_dets = detector.predict(
        resize_crops([crop[0] for crop in target_crops], CONFIG["inputs"]["detector"])
    )
    print(time.perf_counter() - start)

    # Draw the detections
    draw_image(img_path, clf_res, image_dets, [region[1] for region in target_crops])
