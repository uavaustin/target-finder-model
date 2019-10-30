#!usr/bin/env python3
"""
Assumptions: ./data/targets directory already created

This script should create all combinations of shape, alphanumeric, shape color, and alpha color.
Shape color and alpha color come from the config file. While there are all combos of colors in the config file, we don't include all combos of RBG values (just randomized from a few each time). 

In total, this should create about 2.5 GB of data with about 1.3 million targets.
These targets are all oriented at 0 degrees and all the same size.
We'll add rotation and size varients later using targets generated from this script.

"""
import glob
import multiprocessing
import os
import random
import sys
from tqdm import tqdm
import itertools
import functools

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

import createUnrotatedConfig
import helperFunctions

# Get constants from createUnrotatedConfig
SHAPE_TYPES = createUnrotatedConfig.SHAPE_TYPES
ALPHAS = createUnrotatedConfig.ALPHAS
ALPHA_FONTS = createUnrotatedConfig.ALPHA_FONTS
TARGET_COLORS = createUnrotatedConfig.TARGET_COLORS
ALPHA_COLORS = createUnrotatedConfig.ALPHA_COLORS
COLORS = createUnrotatedConfig.COLORS
TARGETS_DIR = createUnrotatedConfig.TARGETS_DIR
DATA_DIR = createUnrotatedConfig.DATA_DIR
UNROTATED_DIR = createUnrotatedConfig.UNROTATED_DIR

def multithreadTargetCreation(listOfTargetParams):
  #print(len(list(listOfTargetParams)))
  with multiprocessing.Pool(None) as pool:
   processes = pool.imap_unordered(_create_one_target, listOfTargetParams)
   for i in tqdm(processes, total = len(listOfTargetParams)):
    pass


def get_zipped_target_params():
  #need SHAPE_TYPES, ALPHAS, ALPHA_FONTS, TARGET_COLORS, ALPHA_COLORS

  allParamsSeperateLists = [SHAPE_TYPES,ALPHAS,ALPHA_FONTS,TARGET_COLORS,ALPHA_COLORS]
  cartesianProductParams = itertools.product(*allParamsSeperateLists)
    
  paramLength = list(map(len,allParamsSeperateLists)) 
  print(paramLength)
  #total = functools.reduce(lambda a,b: a*b, paramLength, 1)
  #print(total)
  #print(list(cartesianProductParams))

  return cartesianProductParams

def _create_one_target(target_params):
    """Create a shape given all the input parameters"""
    (shape, alpha, font_file, target_color, alpha_color) = target_params

    shape_image = helperFunctions._get_base_shapes(shape)

    #so shape and alpha aren't the same color
    if target_color == alpha_color:
      alpha_color = 'white'

    #get rgb values associated with the color and augment it
    target_rgb = random.choice(COLORS[target_color])
    target_rgb = helperFunctions._augment_color(target_rgb)
    alpha_rgb = random.choice(COLORS[alpha_color])
    alpha_rgb = helperFunctions._augment_color(alpha_rgb)
    

    image = helperFunctions._get_base(shape_image, target_rgb)
    image = helperFunctions._strip_image(image)
    image = helperFunctions._add_alphanumeric(image, shape, alpha, alpha_rgb, font_file)


    image = helperFunctions._strip_image(image)
    
    #create string w/ target info to name img when saved
    target_name = create_target_name(shape, alpha, target_color, alpha_color)
    #write the single created target to file system
    write_single_target_img(image, target_name)

    return image


def create_target_name(shape, alpha, target_color, alpha_color):
    return "SH-" + shape + "-AN-" + alpha + "-TC-" + target_color + "-AC-" + alpha_color

def write_single_target_img(image, image_name):
    data_path = os.path.join(DATA_DIR, TARGETS_DIR, UNROTATED_DIR)
    single_target_fn = '{}.png'.format(image_name)
    single_target_path = os.path.join(data_path, single_target_fn)  

    image.save(single_target_path)

if __name__ == '__main__':
  zippedList4Multithreading = list(get_zipped_target_params())
  #print(zippedList4Multithreading)
  #_create_one_target(zippedList4Multithreading[0])
  multithreadTargetCreation(zippedList4Multithreading)
