#!/usr/bin/env python3

from tqdm import tqdm
from PIL import Image
import random
import cropsConfig 
import glob
import os
import multiprocessing
import itertools

# Get constants from cropsConfig
CLF_WIDTH, CLF_HEIGHT = cropsConfig.PRECLF_SIZE
CROP_WIDTH, CROP_HEIGHT = cropsConfig.CROP_SIZE
OVERLAP = cropsConfig.CROP_OVERLAP
RATIO = CLF_WIDTH / CROP_WIDTH
FILE_PATH = os.path.abspath(os.path.dirname(__file__))
CLASSES = cropsConfig.CLF_TYPES
NUM_BIG_BGS = cropsConfig.NUM_BIG_BGS
OFFSET = cropsConfig.OFFSET

def create_the_data_procedure():
  create_clf_data_directories()
  create_list_big_background()

def create_clf_data_directories():
  #name dataset (ex clf_train or clf_val)
  new_dataset = cropsConfig.CROPS_DIR
  #name the new path you would like
  new_cropped_backgrounds_path = os.path.join(cropsConfig.DATA_DIR, new_dataset)
  #create that path you just named 
  os.makedirs(new_cropped_backgrounds_path, exist_ok=True)

def create_list_big_background():
  #gets path string to folder containing big background images (not sure if second arg is good style bc it references 'train' folder in this case, and not 'clf_train' idk)  
  big_backgrounds_path = os.path.join(cropsConfig.DATA_DIR, cropsConfig.ORIGINALS_TO_CROP)
 
 #new version to keep data together instead of writing to and reading from txt files
 #joins each specific big background path in a list
  big_backgrounds_in_list = []
  for i in range(OFFSET, NUM_BIG_BGS + OFFSET):
    #tuple form (big background img path, big background number)
    big_background_path = os.path.join(big_backgrounds_path, f'ex{i}.png')
    background_tuple = (big_background_path, i)
    big_backgrounds_in_list.append(background_tuple)

  data = big_backgrounds_in_list
  with multiprocessing.Pool(None) as pool:
        processes = pool.imap_unordered(create_many_crops_from_big_background, data)
        for i in tqdm(processes, total=NUM_BIG_BGS):
            pass

 

def create_many_crops_from_big_background(big_background_tuple):
  big_background_img = Image.open(big_background_tuple[0])
  big_background_number = big_background_tuple[1]

  full_width, full_height = big_background_img.size
  
  crops_list_from_one_big_background = []
  
  crop_number_from_current_bg = 0

  for y1 in range(0, full_height - CROP_HEIGHT, CROP_HEIGHT - OVERLAP):
    for x1 in range(0, full_width - CROP_WIDTH, CROP_WIDTH - OVERLAP):
      y2 = y1 + CROP_HEIGHT
      x2 = x1 + CROP_WIDTH

      cropped_img = big_background_img.crop((x1, y1, x2, y2))
      cropped_img = cropped_img.resize((CLF_WIDTH, CLF_HEIGHT))
      #named bg(bg_number)_crop(y-coord)-(x-coord), pass in image and name
      write_single_crop_to_file(cropped_img, 'bg{}_crop{}'.format(big_background_number,format(crop_number_from_current_bg, '02d')))
      
      crop_number_from_current_bg += 1

def write_single_crop_to_file(cropped_img, cropped_image_name):
  #path string to folder to store crops
  data_path = os.path.join(cropsConfig.DATA_DIR, cropsConfig.CROPS_DIR)
  single_crop_fn = '{}.png'.format(cropped_image_name)
  single_crop_path = os.path.join(data_path, single_crop_fn)

  cropped_img.save(single_crop_path)


if __name__ == "__main__":
        #set variables DATASET_TYPE, MODEL_TYPE, NUM_BIG_BGS, OFFSET in cropsConfig file
	create_the_data_procedure()

