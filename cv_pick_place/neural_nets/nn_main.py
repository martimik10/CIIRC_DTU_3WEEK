TRAIN =True
from object_detect.run import run
from object_detect.train import train

if(TRAIN):
    train()
else:
    run()