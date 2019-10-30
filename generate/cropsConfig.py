import os

#INSTRUCTIONS: set parameters above #-line in this file. Make sure the original backgrounds you want to crop are in ./data/originalBGs/ You shouldn't have to change anything in the script createCrops.py 

#set dataset type (ie. train or val) (include quotes ex. 'train')
DATASET_TYPE = 'train'

#set model type (ie. clf_ or det_) (include underscore and quotes ex. 'clf_')
MODEL_TYPE = 'clf_'

#set name of folder where original backrounds are that you want cropped
ORIGINALS_TO_CROP = 'originalBGs'

#set folder (as string) you want the crops to be in (will be in data/[folder]) (assumes folder already created)
CROPS_DIR = 'crops'

#set number of original background images you'd like to crop up
NUM_BIG_BGS = 2

#set offset (if you'd like to start from from something other than the first background)
OFFSET = 0

############################

PRECLF_SIZE = (64, 64)

CROP_SIZE = (400, 400)

CLF_TYPES = ['background', 'shape_target']

CROP_OVERLAP = 100

DATA_DIR = os.environ.get('DATA_DIR',os.path.join(os.path.dirname(__file__), 'data'))


