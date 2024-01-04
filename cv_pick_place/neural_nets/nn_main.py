TRAIN =True
from object_detect.run import run
from object_detect.train import train_detect

if(TRAIN):
    train_detect()
else:
    run()