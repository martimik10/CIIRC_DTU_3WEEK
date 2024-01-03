
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

#get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print("curr_dir", current_dir)
print("parent_dir", parentdir)
# sys.path.append('..')
# from ...Dataset import *
from Dataset.GenerateSyntheticDataset import generate_dataset



DATASET_CHECK = False

if(DATASET_CHECK):
    generate_dataset(custom_num_images=10,segmentation=False)
