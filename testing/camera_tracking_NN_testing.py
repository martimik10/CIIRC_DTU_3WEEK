import os
import time 


from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import pandas as pd
import cv2  
from pathlib import Path
from realsense_depth_test import DepthCamera

#constants
RESIZE_WINDOW_FACTOR = 3 #TODO: change to be a function of screen size
CONFIDENCE_THRESHOLD = 0.9
GREEN = (0, 255, 0)
GREY = (211,211,211)
WHITE = (255, 255, 255)
MODEL_TO_USE = "best_detect_08_01.pt" #TODO change to relative when applicable
REALSENSE_CFG_PATH = Path(__file__).parent.parent / "cv_pick_place" / "config" / "D435_camera_config.json"

# get current directory, move above one step, then go to /congig
camera = DepthCamera(config_path=REALSENSE_CFG_PATH)

# Initialize DeepSort tracker

tracker = DeepSort(max_age=10) #TODO: change max age to be a function of fps
# Get frames from camera
def run_NN_on_frame(frame,model):
        ouput = model(frame)
        return ouput

if __name__ == "__main__":
    # MODEL_TO_USE = f'runs/detect/train/weights/best.pt'

    try:
        model = YOLO(MODEL_TO_USE)
        print("[INFO] Model loaded: YoloV8", MODEL_TO_USE)
        
    except FileNotFoundError:
        print("[INFO] Model not found, YoloV8n")

    ################
    # Initialize opencv window
    ################
    # img_array =[]
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (0, 315)
    fontScale = 1
    fontColor = (0, 0, 255)
    thickness = 2
    lineType = 2
    prev_frame_time= 0
    while True:
        
        success,frame_timestamp,depth_frame,rgb_frame,colorized_depth = camera.get_frames()
        if not success:
            continue
        img_frame = rgb_frame.copy()
        img_frame = cv2.resize(img_frame,(640,320))
        
        ######################################
        # Run YOLOv8 inference on the frame
        ######################################
        
        results = []
        detections = model(img_frame)
        
        for data in detections[0].boxes.data.tolist():
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = data[4]

            # filter out weak detections by ensuring the 
            # confidence is greater than the minimum confidence
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            # if the confidence is greater than the minimum confidence,
            # get the bounding box and the class id
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[5])
            # add the bounding box (x, y, w, h), confidence and class id to the results list
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

        ######################################
        # TRACKING
        ######################################
        # update the tracker with the new detections
        tracks = tracker.update_tracks(results, frame=img_frame)
        # loop over the tracks
        for track in tracks:
            # if the track is not confirmed, ignore it
            if not track.is_confirmed():
                print("could not track")
                continue

            # get the track id and the bounding box
            track_id = track.track_id
            ltrb = track.to_ltrb()

            xmin = int(ltrb[0])
            ymin = int(ltrb[1])
            xmax = int(ltrb[2])
            ymax = int(ltrb[3])

            # draw the bounding box and the track id
            
            cv2.rectangle(img_frame, (xmin, ymin), (xmax, ymax), color=GREY, thickness=1)
            cv2.rectangle(img_frame, (xmin, ymin - 20), (xmin + 20, ymin), color=GREY, thickness=-1)
            cv2.putText(img_frame, str(track_id), (xmin + 5, ymin - 8),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=WHITE, thickness=1)
            
    
        
        ######################################
        # DISPLAY
        ######################################
        annotated_frame = detections[0].plot()
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))
        cv2.putText(annotated_frame, 'FPS: ' + fps,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

        # Display the annotated frame
        #resize window to fit screen
        cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLOv8 Inference", 640*RESIZE_WINDOW_FACTOR, 320*RESIZE_WINDOW_FACTOR)
        
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        # img_array.append(annotated_frame)


        # Break the loop if 'q', 'ESC' or Close button is pressed
        key = cv2.waitKey(1)

        if key == ord("q") or key == 27 or cv2.getWindowProperty("YOLOv8 Inference", cv2.WND_PROP_VISIBLE) < 1:
            print("[INFO] Stopping YOLOv8 Inference")
            break

        
    cv2.destroyAllWindows()
