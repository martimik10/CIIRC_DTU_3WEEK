
import time 
import os
import platform

from ultralytics import YOLO
import torch


from Dataset.dataset_check import dataset_check
from Dataset.GenerateSyntheticDataset import generate_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 30
GENERATE_NR_IMAGES = 220


def train(mode = "Detect"):
        # Load already trained weights
        
        ###########
        if(dataset_check()):
             generate_dataset(GENERATE_NR_IMAGES,segmentation=mode=="Segment")
       
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
        results = model.train(data='generated_dataset_Win.yaml', epochs=EPOCHS)
        # else:
        #     print('Using Linux')
        #     results = model.train(data='generated_dataset.yaml', epochs=EPOCHS)
        end = time.time()
        
        # Save the results
        results['Epochs'] = EPOCHS
        results['Time training'] = end - start