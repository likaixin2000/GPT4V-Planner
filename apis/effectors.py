###
# conda create --name contact_graspnet python=3.8
# conda activate contact_graspnet
# conda install -c conda-forge cudatoolkit=11.2(or 11.3)
# conda install -c conda-forge cudnn=8.2
# pip install tensorflow==2.5 tensorflow-gpu=2.5
# pip install opencv-python-headless
# pip install pyyaml==5.4.1
# pip install pyrender
# pip install tqdm
# pip install mayavi
# pip install opencv-python==4.2.0.34
# conda install libffi==3.3
###
# import rospy
import cv2
import numpy as np

from service.grasper.grasper import FetchAgent


class Grasper:
    def __init__(self):
        self._fetch_agent = None

    def setup(self):
        self._fetch_agent = FetchAgent()
        print("Finished setting up the fetch agent.")

    def pick(self, mask):
        pass

    def grasp(self, mask):
        pass

    def move(self, name):
        pass
