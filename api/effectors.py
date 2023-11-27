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
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from common_service.srv import GraspGroup

# Intrinsic camera matrix
K = [541.7874545077664, 0.0, 321.6009893833978, 0.0, 538.1475661088576, 232.013175647453, 0.0, 0.0, 1.0]

class Effector:
    def __init__(self):
        rospy.init_node('grasping_node')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/head_camera/rgb/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/head_camera/depth_registered/image_raw', Image, self.depth_callback)
        self.rgb_image = None
        self.depth_image = None

    def image_callback(self, msg):
        self.rgb_image = msg

    def depth_callback(self, msg):
        self.depth_image = msg

    def grasp(self, mask):
        seg_msg = self.bridge.cv2_to_imgmsg(mask.astype(np.uint8), encoding="mono8")
        rospy.wait_for_service('contact_graspnet/get_grasp_result')
        grasping_service = rospy.ServiceProxy('contact_graspnet/get_grasp_result', GraspGroup)
        
        segmap_id = 1
        poses = grasping_service(self.rgb_image, self.depth_image, seg_msg, K, segmap_id)
        return poses.grasp_poses
        