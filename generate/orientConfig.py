import os

#folder where the (cropped) images you'd like oriented
IMGS_TO_BE_CROPPED_DIR = 'crops'

DATA_DIR = os.environ.get('DATA_DIR',os.path.join(os.path.dirname(__file__), 'data'))

CROPS_DIR = os.path.join(DATA_DIR, IMGS_TO_BE_CROPPED_DIR) 

ORIENT_DIR = os.path.join(DATA_DIR, 'orient')

