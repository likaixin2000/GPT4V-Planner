import os
import sys
import numpy as np
import rospy
from sensor_msgs.msg import Image
from gsam.srv import PerceptionService, PerceptionPointService
from cv_bridge import CvBridge
import tf
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from common_service.srv import GraspGroup, StringService

from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion
import math
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Point, Point32
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import re
import time
# sys.path.append('/home/anxing/work/catkin_ws/src/gsam/src/Grounded_Segment_Anything')
# sys.path.append('/home/anxing/work/catkin_ws/src/gsam/src/Grounded_Segment_Anything/Tag2Text')


class FetchAgent:
    def __init__(self) -> None:
        rospy.init_node('fetch_agent')
        self.is_init = False
        self.max_distance_for_perception = 4 ## only detect the objects in 4 meters range
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.target = None
        self.map_data = None
        self.amcl_pose = None
        self.current_angle = None
        self.object_list = []
        self.known_locations = [
            ('living room', -0.9023915518434525, -0.7027813214394424, -0.015823050286178413, 0.999874807703265),
            ('projector', -1.276164051454064, 4.120375689830889, 0.3780486827141957, 0.9257857168362781),
            ('kitchen', -2.7854025716787314, 0.5172084230749142, 0.7302201123857227, 0.6832119637911667), #0.8451152423109035, 0.5345841628909173
            ('tv', -0.89181461293163, -5.829970065859149, -0.707611870273355, 0.7066013310546794),
            ('sofa', 0.05036759855253443,-2.55771778357706,-0.9990183854832045, 0.04429746568971257),
            ('main door', 0.9701700664599943, -1.435972057053516, -0.0050052644906170484, 0.999987473585234),
            ('fridge', -3.8819888151714506, 3.1554459303191456, 0.8610737531288837, 0.5084800799171372),
            ('lamp', -2.1745207202108108, -1.4043145612539867, -0.7449554438018537, 0.6671142231657058),
            ('table', -0.02831879749484199, 1.0933972715123002, 0.6984894736475025, 0.7156203289479939),
            # ('pre_table', -0.{004935663953263684, 0.761515023759, 0.7110862790969399, 0.7031047601034065)

        ]
        self.currentRoomName = "livingroom1"
        self.is_kitchencountertop = True
        self.visited = []
        self.visited.append(self.currentRoomName)
        # Subscribe to RGB and Depth images
        self.image_sub = rospy.Subscriber('/head_camera/rgb/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/head_camera/depth_registered/image_raw', Image, self.depth_callback)
        self.global_costmap_sub = rospy.Subscriber('/move_base/global_costmap/costmap', OccupancyGrid, self.global_costmap_callback)


        self.marker_pub = rospy.Publisher('/detected_objects_markers', MarkerArray, queue_size=10)
        
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.pose_callback)
        
        self.head_traj_client = actionlib.SimpleActionClient('head_controller/follow_joint_trajectory',
                                                             FollowJointTrajectoryAction)
        self.head_traj_client.wait_for_server()

        self.torso_traj_client = actionlib.SimpleActionClient('torso_controller/follow_joint_trajectory',
                                                              FollowJointTrajectoryAction)
        self.torso_traj_client.wait_for_server()

        self.action_service = rospy.ServiceProxy('exe_task', StringService)    
            # Call the service
        # response = perception_service(self.rgb_image, self.depth_image)
        
        self.tf_listener = tf.TransformListener()
        self.prepare()

    def process_mask(self, mask):
        new_mask = []
        for pt in mask:
            x = round(pt[1])
            y = round(pt[0])
            new_mask.append([x,y])
        return new_mask

    def transform_format(self, s):
        # Remove characters inside and including parentheses
        s = ''.join(s.split('(')[0:-1])

        # Remove spaces and return
        return s.replace(' ', '')

## 0.36330658197402954, [0, 0.1928356905883789]
    def move_torso_head(self, torso_position, head_position, duration=1.0):
        head_joints = ["head_pan_joint", "head_tilt_joint"]
        torso_joints = ["torso_lift_joint"]

        head_goal = FollowJointTrajectoryGoal()
        head_goal.trajectory = JointTrajectory()
        head_goal.trajectory.joint_names = head_joints

        torso_goal = FollowJointTrajectoryGoal()
        torso_goal.trajectory = JointTrajectory()
        torso_goal.trajectory.joint_names = torso_joints

        head_point = JointTrajectoryPoint()
        head_point.positions = tuple(head_position)
        head_point.time_from_start = rospy.Duration(duration)
        head_goal.trajectory.points.append(head_point)

        torso_point = JointTrajectoryPoint()
        torso_point.positions = (torso_position,)
        torso_point.time_from_start = rospy.Duration(duration)
        torso_goal.trajectory.points.append(torso_point)

        self.head_traj_client.send_goal(head_goal)
        self.head_traj_client.wait_for_result()
        self.torso_traj_client.send_goal(torso_goal)
        self.torso_traj_client.wait_for_result()

    def prepare(self):
        torso = 0.3
        head = [0, 0.5]
        self.move_torso_head(torso, head)
        

    def get_observation(self):
        object_name_list = [obj[0] for obj in self.object_list]
        room_name_list = [obj[0] for obj in self.room_list]

        return "Object List: " + str(
            object_name_list) + "\nRoom List: " + str(
            room_name_list) + "\nCurrent Room: " + str(self.currentRoomName)



    def map_callback(self, data):
        self.map_data = data

    def pose_callback(self, data):
        self.amcl_pose = data
        orientation_q = data.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, self.current_angle) = euler_from_quaternion(orientation_list)

    def image_callback(self, msg):
        self.rgb_image = msg

    def depth_callback(self, msg):
        self.depth_image = msg

    def global_costmap_callback(self, data):
        self.global_costmap_data = data


    def get_closest_free_space(self, pose):
        if not self.global_costmap_data:
            return None

        # Convert the object's position to map coordinates
        map_x = int((pose.position.x - self.global_costmap_data.info.origin.position.x) / self.global_costmap_data.info.resolution)
        map_y = int((pose.position.y - self.global_costmap_data.info.origin.position.y) / self.global_costmap_data.info.resolution)

        # Iterate surrounding cells for a given range (e.g., 10 cells)
        search_radius = 30
        closest_free_space = None
        min_distance = float('inf')

        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                cell_x = map_x + dx
                cell_y = map_y + dy

                index = cell_y * self.global_costmap_data.info.width + cell_x
                if 0 <= index < len(self.global_costmap_data.data) and self.global_costmap_data.data[index] == 0:  # Free space
                    # Compute Euclidean distance
                    distance = ((dx * self.global_costmap_data.info.resolution)**2 + (dy * self.global_costmap_data.info.resolution)**2)**0.5
                    if distance < min_distance:
                        min_distance = distance
                        closest_free_space = (cell_x * self.global_costmap_data.info.resolution + self.global_costmap_data.info.origin.position.x, 
                                            cell_y * self.global_costmap_data.info.resolution + self.global_costmap_data.info.origin.position.y)

        if closest_free_space:
            # Calculate yaw angle
            delta_x = pose.position.x - closest_free_space[0]
            delta_y = pose.position.y - closest_free_space[1]
            yaw = math.atan2(delta_y, delta_x)

            return closest_free_space, yaw
        else:
            return None, None


    def format_object_names(self, object_list):
        """
        Format a list of objects (in tuple form) by appending a unique number to repeated object names.

        Args:
        - object_list (list of tuple): List of object names in tuple form.

        Returns:
        - list of tuple: List of formatted object tuples.
        """

        name_counts = {}
        formatted_objects = []
        # microwave_name = ['appliancemicrowaveoven,appliancemicrowave']

        for obj in object_list:
            name, *properties = obj
            name = self.extract_base_name(name)
            if 'microwave' in name:
                name = 'microwave'
            if name == 'appliance':
                name = 'microwave'
            if 'counter' in name:
                name = 'kitchencounter'
            if 'food' in name:
                name = 'food'
            if name not in name_counts:
                name_counts[name] = 1
            else:
                name_counts[name] += 1

            formatted_name = f"{name}{name_counts[name]}"
            formatted_objects.append((formatted_name, *properties))

        return formatted_objects
    
    def extract_base_name(self, name):
        """Extract the base name by removing trailing digits."""
        base_name = ''.join([char for char in name if not char.isdigit()])
        return base_name

    def filter(self, objects):

        filter_list = ['livingroom', 'meetingroom', 'kitchen', 'officeroom',"bathroom", "room"]
        # Merge detected objects based on their name and position
        filtered_objects = []
        for obj in objects:
            # Search merged_objects for an object with the same name and similar position
            obj_base_name = self.extract_base_name(obj[0])

            if not any(filter_item in obj_base_name for filter_item in filter_list):
                filtered_objects.append(obj)
            # if obj_base_name not in filter_list:
            #     filtered_objects.append(obj)

        # Update the object list with the merged objects

        return filtered_objects
    
    def merge_objects(self, all_detected_objects):

        threshold_distance = 0.5  # Adjust based on requirements

        # Merge detected objects based on their name and position
        merged_objects = []
        for obj in all_detected_objects:
            # Search merged_objects for an object with the same name and similar position
            obj_base_name = self.extract_base_name(obj[0])
            # Search merged_objects for an object with the same base name and similar position
            existing_obj = next((x for x in merged_objects if self.extract_base_name(x[0]) == obj_base_name and self.distance_between_points(x[1:3], obj[1:3]) < threshold_distance), None)
            if not existing_obj:
                merged_objects.append(obj)

        # Update the object list with the merged objects
        self.object_list = merged_objects

        return merged_objects

    def distance_between_points(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    

    def update(self):
        pass

    def move(self, target_name):
        print("move to "+ target_name)
        # return 
        # Find the object in the list
        # if (self.is_init == False):
            
        if self.is_init == True:
            rospy.wait_for_service('grounded_manipulation_service/init')
            init_service = rospy.ServiceProxy('grounded_manipulation_service/init', StringService)
            s = "end"
            response = init_service(s)
            self.is_init = False

        for name, x, y, q2, q3 in self.known_locations:
            if target_name == name:
                client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
                client.wait_for_server()

                goal = MoveBaseGoal()
                goal.target_pose.header.frame_id = "map"
                goal.target_pose.header.stamp = rospy.Time.now()
                
                goal.target_pose.pose.position.x = x
                goal.target_pose.pose.position.y = y
                goal.target_pose.pose.position.z = 0.0  # Assuming z is always 0 for simplicity
                
                goal.target_pose.pose.orientation.x = 0
                goal.target_pose.pose.orientation.y = 0
                goal.target_pose.pose.orientation.z = q2
                goal.target_pose.pose.orientation.w = q3
                
                client.send_goal(goal)
                wait = client.wait_for_result()
                self.currentRoomName = target_name

                ## prepare and scan the env
                # if target_name not in self.visited:
                #     self.visited.append(target_name)
                #     self.prepare()
                #     self.rotate_and_scan()

                if not wait:
                    rospy.logerr("Action server not available!")
                else:
                    return 
                return 


    def rotate(self, target_yaw_angle):
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        client.wait_for_server()

        # get the current orientation
        current_orientation = self.amcl_pose.pose.pose.orientation
        current_yaw = tf.transformations.euler_from_quaternion([
            current_orientation.x,
            current_orientation.y,
            current_orientation.z,
            current_orientation.w
        ])[2]  # yaw

        # calculate target yaw
        target_yaw = target_yaw_angle
        target_orientation = tf.transformations.quaternion_from_euler(0, 0, target_yaw)

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        goal.target_pose.pose.position = self.amcl_pose.pose.pose.position
        goal.target_pose.pose.orientation.x = target_orientation[0]
        goal.target_pose.pose.orientation.y = target_orientation[1]
        goal.target_pose.pose.orientation.z = target_orientation[2]
        goal.target_pose.pose.orientation.w = target_orientation[3]

        client.send_goal(goal)
        wait = client.wait_for_result()

        if not wait:
            rospy.logerr("Action server not available!")
            return False

        return True


    def find_closest_location(self):
        cx = self.amcl_pose.pose.pose.position.x
        cy = self.amcl_pose.pose.pose.position.y

        # fridge, counter, microwave
        # loc_list = np.array([[-4.0856,3.8],[-4.0056,3.3],[-3.933, 2.8538]])
        loc_list = np.array([[-4.01,3.85],[-4.156,3.5],[-4.156, 2.9]])

        # Calculate the squared distances
        distances = np.sum((loc_list - [cx, cy])**2, axis=1)
        
        # Find the index of the minimum distance
        index_min_distance = np.argmin(distances)
        
        # Return the closest location
        return index_min_distance

    def pick(self, mask):
        if self.is_init == False:
            rospy.wait_for_service('grounded_manipulation_service/init')
            init_service = rospy.ServiceProxy('grounded_manipulation_service/init', StringService)
            s = "start"
            response = init_service(s)
            self.is_init = True

        rospy.wait_for_service('contact_graspnet/get_grasp_result')

        grasping_service = rospy.ServiceProxy('contact_graspnet/get_grasp_result', GraspGroup)

        mask = np.array(mask, dtype=np.uint8)  # Convert to uint8 if not already

        # Convert the image to sensor_msgs/Image
        mask = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")

        K = [541.7874545077664, 0.0, 321.6009893833978, 0.0, 538.1475661088576, 232.013175647453, 0.0, 0.0, 1.0]
        poses = grasping_service(self.rgb_image, self.depth_image, mask, K, segmap_id=1)
        
        return poses.grasp_poses

    def placeon(self, mask):
        if self.is_init == False:
            rospy.wait_for_service('grounded_manipulation_service/init')
            init_service = rospy.ServiceProxy('grounded_manipulation_service/init', StringService)
            s = "start"
            response = init_service(s)
            self.is_init = True

        rospy.wait_for_service('grounded_manipulation_service/place')

        grasping_service = rospy.ServiceProxy('grounded_manipulation_service/place', GraspGroup)

        mask = np.array(mask, dtype=np.uint8)  # Convert to uint8 if not already

        # Convert the image to sensor_msgs/Image
        mask = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")

        K = [541.7874545077664, 0.0, 321.6009893833978, 0.0, 538.1475661088576, 232.013175647453, 0.0, 0.0, 1.0]
        poses = grasping_service(self.rgb_image, self.depth_image, mask, K, segmap_id=1)
        
        return poses.grasp_poses

    def open(self, obj_name):
        print("open "+ obj_name)


        if "fridge" in obj_name:
            task_name = "open_fridge"
        elif "microwave" in obj_name:
            task_name = "open_microwave"

        response = self.action_service(task_name)
        time.sleep(10)

        # time.sleep(13)
        
        return


        # input_name = input("Press Enter to finish or enter Fail \n")
        # if not input_name:  # if user just pressed Enter
        #                 ## prepare and scan the env
        #     self.prepare()
        #     self.scan()
        #     return
        # elif input_name=="Fail":
        #     return "Fail"


    def close(self, obj_name):
        print("close "+obj_name)

        # if "fridge" in obj_name:
        #     task_name = "close_fridge"
        #     response = self.action_service(task_name)

        return

        # input_name = input("Press Enter to finish or enter Fail \n")
        # if not input_name:  # if user just pressed Enter
        #     return
        # elif input_name=="Fail":
        #     return "Fail"


    def switchon(self, obj_name):
        print("switch on "+obj_name)

        if "microwave" in obj_name:
            task_name = "switchon_microwave"

        response = self.action_service(task_name)
        return
    
        # input_name = input("Press Enter to finish or enter Fail \n")
        # if not input_name:  # if user just pressed Enter
        #     return
        # elif input_name=="Fail":
        #     return "Fail"


    def vqa(self, qusetion):
        print("Ask the question: "+ qusetion)
        rospy.wait_for_service('vlm_service')

        vlm_service = rospy.ServiceProxy('vlm_service', StringService)
        return vlm_service(data = "You are a robot butler in singapore. Answer this question in 20 words:" + qusetion).response

        # return

    def check(self, qusetion):
        print("Check the task: "+ qusetion)

        return True

    def scan(self):
        # self.object_list = []

        if not self.rgb_image or not self.depth_image:
            print("No images received yet!")
            return
        
        # Wait for the service to become available
        rospy.wait_for_service('perception_service')
        
        try:
            print("start perception")
            # Create a handle to the service
            perception_service = rospy.ServiceProxy('perception_service', PerceptionService)
            
            # Call the service
            response = perception_service(self.rgb_image, self.depth_image)
            
            # Print the results
            # print("Object Labels: ", response.object_labels)

            poses = [(response.x_positions[i],response.y_positions[i],response.theta_angles[i]) for i in range(len(response.object_labels))]
            # print("Poses of Object: ", poses)

            marker_array = MarkerArray()
            self.marker_id_counter = 0
            for idx, (x, y, theta) in enumerate(poses):
                object_pose_in_robot_frame = PoseStamped()
                object_pose_in_robot_frame.header.frame_id = "head_camera_link"  # Assuming the robot's main frame is called base_link
                object_pose_in_robot_frame.pose.position.x = x
                object_pose_in_robot_frame.pose.position.y = y
                quat = tf.transformations.quaternion_from_euler(0, 0, theta)
                object_pose_in_robot_frame.pose.orientation.x = quat[0]
                object_pose_in_robot_frame.pose.orientation.y = quat[1]
                object_pose_in_robot_frame.pose.orientation.z = quat[2]
                object_pose_in_robot_frame.pose.orientation.w = quat[3]

                try:
                    time = self.tf_listener.getLatestCommonTime("map", "head_camera_link")
                    object_pose_in_map_frame = self.tf_listener.transformPose("map", object_pose_in_robot_frame)

                    marker = Marker()
                    marker.header.frame_id = "map"
                    marker.header.stamp = rospy.Time.now()  # Current time
                    marker.id = self.marker_id_counter
                    self.marker_id_counter += 1
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    marker.scale.x = 0.1
                    marker.scale.y = 0.1
                    marker.scale.z = 0.1
                    marker.color.a = 1.0
                    marker.color.r = 1.0
                    marker.pose.position.x = object_pose_in_map_frame.pose.position.x
                    marker.pose.position.y = object_pose_in_map_frame.pose.position.y
                    marker.pose.position.z = 0
                    marker.pose.orientation = object_pose_in_map_frame.pose.orientation

                    free_space, yaw = self.get_closest_free_space(object_pose_in_map_frame.pose)
                    # print(object_pose_in_map_frame.pose)
                    # print(free_space, yaw)
                    if(x**2+y**2 <= self.max_distance_for_perception**2):
                        self.object_list.append((response.object_labels[idx], free_space[0], free_space[1], yaw))
                    # print(self.object_list)
                    
                    marker_array.markers.append(marker)

                    text_marker = Marker()
                    text_marker.header.frame_id = "map"
                    text_marker.header.stamp = rospy.Time.now()
                    text_marker.id = self.marker_id_counter  # Make sure each text marker has a unique id
                    self.marker_id_counter += 1
                    text_marker.type = Marker.TEXT_VIEW_FACING
                    text_marker.action = Marker.ADD
                    text_marker.scale.z = 0.2  # Height of the text, adjust as needed
                    text_marker.color.a = 1.0
                    text_marker.color.r = 1.0
                    text_marker.color.g = 1.0
                    text_marker.color.b = 1.0
                    # text_marker.pose.position.x = object_pose_in_map_frame.pose.position.x
                    # text_marker.pose.position.y = object_pose_in_map_frame.pose.position.y
                    text_marker.pose.position.x = free_space[0]
                    text_marker.pose.position.y = free_space[1]
                    text_marker.pose.position.z = 0.6
                    text_marker.pose.orientation = object_pose_in_map_frame.pose.orientation
                    text_marker.text = response.object_labels[idx]  # Assuming object_labels is a list of strings
                    marker_array.markers.append(text_marker)



                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                    print(e)

            self.marker_pub.publish(marker_array)
            # print("Object list: ", response.object_labels)
            self.object_list = self.filter(self.object_list)

            self.object_list = self.merge_objects(self.object_list)
            self.object_list = self.format_object_names(self.object_list)
 
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def rotate_and_scan(self):
        turn_times = 6
        turn_angle = 60 * math.pi / 180  # Convert 60 degrees to radians
        target_angles = []
        for i in range(turn_times):
            target_angle = self.current_angle - (i+1)*turn_angle 
            if target_angle > math.pi:
                target_angle -= 2 * math.pi
            target_angles.append(target_angle)

        # self.object_list = []

        for target_angle in target_angles:



            if not self.rgb_image or not self.depth_image:
                print("No images received yet!")
                return
            
            # Wait for the service to become available
            rospy.wait_for_service('perception_service')
            
            try:
                print("start perception")
                # Create a handle to the service
                perception_service = rospy.ServiceProxy('perception_service', PerceptionService)
                
                # Call the service
                response = perception_service(self.rgb_image, self.depth_image)
                
                # Print the results
                # print("Object Labels: ", response.object_labels)

                poses = [(response.x_positions[i],response.y_positions[i],response.theta_angles[i]) for i in range(len(response.object_labels))]
                # print("Poses of Object: ", poses)
                # poses = [x for x in poses if x[0]**2+x[1]**2 <= 25]

                marker_array = MarkerArray()
                self.marker_id_counter = 0
                for idx, (x, y, theta) in enumerate(poses):
                    if x**2 + y**2 <= 16:
                        # [x for x in poses if x[0]**2+x[1]**2 <= 25]
                        object_pose_in_robot_frame = PoseStamped()
                        object_pose_in_robot_frame.header.frame_id = "head_camera_link"  # Assuming the robot's main frame is called base_link
                        object_pose_in_robot_frame.pose.position.x = x
                        object_pose_in_robot_frame.pose.position.y = y
                        quat = tf.transformations.quaternion_from_euler(0, 0, theta)
                        object_pose_in_robot_frame.pose.orientation.x = quat[0]
                        object_pose_in_robot_frame.pose.orientation.y = quat[1]
                        object_pose_in_robot_frame.pose.orientation.z = quat[2]
                        object_pose_in_robot_frame.pose.orientation.w = quat[3]

                        try:
                            time = self.tf_listener.getLatestCommonTime("map", "head_camera_link")
                            object_pose_in_map_frame = self.tf_listener.transformPose("map", object_pose_in_robot_frame)

                            marker = Marker()
                            marker.header.frame_id = "map"
                            marker.header.stamp = rospy.Time.now()  # Current time
                            marker.id = self.marker_id_counter
                            self.marker_id_counter += 1
                            marker.type = Marker.CUBE
                            marker.action = Marker.ADD
                            marker.scale.x = 0.1
                            marker.scale.y = 0.1
                            marker.scale.z = 0.1
                            marker.color.a = 1.0
                            marker.color.r = 1.0
                            marker.pose.position.x = object_pose_in_map_frame.pose.position.x
                            marker.pose.position.y = object_pose_in_map_frame.pose.position.y
                            marker.pose.position.z = 0
                            marker.pose.orientation = object_pose_in_map_frame.pose.orientation

                            free_space, yaw = self.get_closest_free_space(object_pose_in_map_frame.pose)
                            # print(object_pose_in_map_frame.pose)
                            # print(free_space, yaw)
                            if(free_space and (x**2+y**2 <= self.max_distance_for_perception**2)):
                                self.object_list.append((response.object_labels[idx], free_space[0], free_space[1], yaw))
                            


                            # print(self.object_list)
                            


                        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                            print(e)

                # print("Object list: ", [x[0] for x in self.object_list])

                # print("Object list: ", response.object_labels)

            except rospy.ServiceException as e:
                print("Service call failed: %s" % e)

            self.rotate(target_angle)  # rotate 60 degrees
        
        # if self.is_kitchencountertop and self.currentRoomName == "kitchen1":
        #     self.object_list.append(("kitchencountertop1", -4.00, 3.515325, 3.10))

        self.object_list = self.filter(self.object_list)
        self.object_list = self.merge_objects(self.object_list)
        self.object_list = self.format_object_names(self.object_list)
        # print("Final object list: ", [x[0] for x in self.object_list])


    def wait(self, animation=True):
        pass

    def get_img(self):
        image = self.bridge.imgmsg_to_cv2(self.rgb_image, 'bgr8')
        return image
