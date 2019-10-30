#assumes /data/all_orientations_crops/images already made (orientConfig.ORIENT_DIR)

import glob
import os
import orientConfig
import PIL
from PIL import Image
import multiprocessing 
from tqdm import tqdm
import itertools

def get_all_crops_in_list():
  dir_with_crops = os.path.join(orientConfig.CROPS_DIR, "*.png")
  crops_path_list = glob.glob(dir_with_crops)
  return crops_path_list

def orient_all_in_list(crops_to_orient):
  with multiprocessing.Pool(None) as pool:
    processes = pool.imap_unordered(orient_4_directions, crops_to_orient)
    for i in tqdm(processes, total=len(crops_to_orient)):
      pass

#orient crop and call write_file
def orient_4_directions(image_path):
  #get image name without ext (/location/file.ext -> file)
  image_name = get_img_name(image_path)

  image_obj = Image.open(image_path)
  #rotate180, mirror, and mirror-rotate180 and save all these with the original with O,R,M, or MR added to file name (original, rotated, mirror, mirror-rotated)
 
  write_oriented_crop(image_obj, image_name, 'O')  

  rotated_image = image_obj.rotate(180)
  write_oriented_crop(rotated_image, image_name, 'R')

  mirrored_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
  write_oriented_crop(mirrored_image, image_name, 'M') 
 
  MR_image = mirrored_image.rotate(180)
  write_oriented_crop(MR_image, image_name, 'MR') 

def get_img_name(image_path):
  cropped_fn = os.path.basename(image_path)
  cropped_fn_wo_ext = os.path.splitext(cropped_fn)[0]
  return cropped_fn_wo_ext

def write_oriented_crop(oriented_img, image_name, orientation):
  #path string to folder to store crops
  orient_path = orientConfig.ORIENT_DIR

  oriented_crop_fn = '{}-{}.png'.format(image_name, orientation)
  oriented_crop_path = os.path.join(orient_path, oriented_crop_fn)

  oriented_img.save(oriented_crop_path)

if __name__ == '__main__':
  crops_list = get_all_crops_in_list()
  orient_all_in_list(crops_list)
