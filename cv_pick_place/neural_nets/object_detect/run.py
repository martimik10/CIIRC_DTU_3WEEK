import os

from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator

def choose_model(runs_dir:str)->str:
    """
     return latest run from directory
    """
    dirs= list(os.walk(runs_dir))
    
    return dirs[0][1][-1]






def run():
      # ultralytics.yolo.utils.plotting is deprecated

        model_to_use=model_to_use = f'runs/detect/{choose_model("runs/detect")}/weights/best.pt'
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)

        try:
                model = YOLO(model_to_use)
        except FileNotFoundError:
                print("Model not found, retraining from scratch with YoloV8n")




        while True:
            _, img = cap.read()
            
            # BGR to RGB conversion is performed under the hood
            # see: https://github.com/ultralytics/ultralytics/issues/2575
            results = model.predict(img)

            for r in results:
                
                annotator = Annotator(img)
                
                boxes = r.boxes
                for box in boxes:
                    
                    b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                    c = box.cls
                    annotator.box_label(b, model.names[int(c)])
                
            img = annotator.result()  
            cv2.imshow('YOLO V8 Detection', img)     
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

        cap.release()
        cv2.destroyAllWindows()

