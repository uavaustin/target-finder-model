# this takes in all unrotated targets in data/targets/all_unrotated_targets dir and rotates each of them 8 times to a random degree (each target will have a rotated copy for all 8, a random degree in (-22.5 - 22.5) (22.5 - 67.5) (67.5 - 112.5) etc... to 360) Then we will save that rotated target in data/targets/rotated_targets with the original name appended with the class of rotation (0,45,90,etc) and not the precise degree of rotation. These params are specified in the config file. Assumes folders already created

# assumes unrotated targets are .png files

import random
import glob
import createUnrotatedConfig
import createRotatedConfig 
import os
from PIL import Image

#set variables (all set in createUnrotatedConfig file)
TARGETS_DIR = createUnrotatedConfig.TARGETS_DIR
DATA_DIR = createUnrotatedConfig.DATA_DIR
UNROTATED_DIR = createUnrotatedConfig.UNROTATED_DIR
ROTATED_DIR = createUnrotatedConfig.ROTATED_DIR

#variables in createRotatedConfig
# this is a dictionary ex. { 0 : (-23, 22), 45 : ...}
ROT_CLASSES = createRotatedConfig.ROT_CLASSES

#TODO create function to multithread unrotated_targets_paths_in_list with rotating function and call it in main

def get_unrotated_target_paths_in_list():
  unrotated_dir_path = os.path.join(DATA_DIR,TARGETS_DIR,UNROTATED_DIR,'*.png')
  all_unrotated_targets_path_list = glob.glob(unrotated_dir_path)
  print(all_unrotated_targets_path_list)
  return all_unrotated_targets_path_list

def rand_rotate_one_target_each_class_and_save(unrotated_target_path):
  #TODO (use dictionary in second config file)
  
  for degree_class in ROT_CLASSES.keys():
    lower_bound, upper_bound = ROT_CLASSES[degree_class]
    interval_set = [x % 360 for x in range(lower_bound, upper_bound)]
    print(interval_set)
    degree_actual = random.choice(interval_set)
    print(degree_actual)
    #rotates target to random degree in current degree_class (ie for class 45 anywhere between 23 and 67 degrees) this returns an open image
    target_image = Image.open(unrotated_target_path)
    rotated_target = target_image.rotate(degree_actual)
    rot_target_fn = name_file(unrotated_target_path, degree_class)
    rotated_target.save(rot_target_fn)


#TODO
def name_file(orig_path, degree_class):
  data_path = os.path.join(DATA_DIR, TARGETS_DIR, ROTATED_DIR)
  orig_target_name = get_img_name(orig_path)
  new_target_name = '{}-D-{}.png'.format(orig_target_name, degree_class)
  file_path_amended_name = os.path.join(data_path, new_target_name)
  return file_path_amended_name

def get_img_name(image_path):
  cropped_fn = os.path.basename(image_path)
  cropped_fn_wo_ext = os.path.splitext(cropped_fn)[0]
  return cropped_fn_wo_ext


if __name__ == "__main__":
  list = get_unrotated_target_paths_in_list()
  rand_rotate_one_target_each_class_and_save(list[0])
