
import time 
import os
import platform
from pathlib import Path
import shutil

from ultralytics import YOLO
import torch


from dataset.dataset_check import dataset_check
from dataset.GenerateSyntheticDataset import generate_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 30
GENERATE_NR_IMAGES = 10


def train(mode = "Segment",yaml_data = Path("cv_pick_place/neural_nets/dataset/GeneratedDataset/data.yaml")):
        # Load already trained weights
        
        ###########
        if(dataset_check()):
             generate_dataset(GENERATE_NR_IMAGES,segmentation=mode=="Segment")
             shutil.copy("cv_pick_place/neural_nets/dataset/data.yaml", "cv_pick_place/neural_nets/dataset/GeneratedDataset/")
        model_to_use = 'runs/detect/train5/weights/best.pt'  # use 'yolov8n.pt' to start fresh
        ###########
        
        #if windows put "/" before model_to_use 
        # curr_os = platform.system()
        # if curr_os == 'Windows':
        #     model_to_use = os.path.join("/",model_to_use)
        
        try:
            model = YOLO(model_to_use)
        except FileNotFoundError:
            print("Model not found, retraining from scratch with YoloV8n")
            model = YOLO('yolov8n.pt')
        
        #check if using cuda
        if torch.cuda.is_available():
            print("Using GPU")
            model.to(device)
        
        
        start = time.time()
        # if curr_os == 'Windows':
        #     print('Using Windows')
        results = model.train(data=yaml_data, epochs=EPOCHS)
        # else:
        #     print('Using Linux')
        #     results = model.train(data='generated_dataset.yaml', epochs=EPOCHS)
        end = time.time()
        
        # Save the results
        results['Epochs'] = EPOCHS
        results['Time training'] = end - start