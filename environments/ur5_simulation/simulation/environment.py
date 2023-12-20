#!/usr/bin/env python

import os
import time
import threading
import pkg_resources

import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from grippers import Suction
import simulation_utils as utils
import cameras


class Environment():

    def __init__(self, disp=False, hz=480):
        """Creates OpenAI gym-style env with support for PyBullet threading.

        Args:
            disp: Whether or not to use PyBullet's built-in display viewer.
                Use this either for local inspection of PyBullet, or when
                using any soft body (cloth or bags), because PyBullet's
                TinyRenderer graphics (used if disp=False) will make soft
                bodies invisible.
            hz: Parameter used in PyBullet to control the number of physics
                simulation steps. Higher values lead to more accurate physics
                at the cost of slower computaiton time. By default, PyBullet
                uses 240, but for soft bodies we need this to be at least 480
                to avoid cloth intersecting with the plane.
        """
        self.ee = None
        self.running = False

        
        self.pixel_size = cameras.RealSenseD415.pixel_size
        self.camera_config = cameras.RealSenseD415.CONFIG
        self.camera_config_up = cameras.RealSenseD415.CONFIG_UP

        self.homej = np.array([-0.65, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        #self.homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        
        self.t_lim = 15 # Set default movej timeout limit.
        self.hz = hz
        self.colors = [utils.COLORS['blue']]

        # Start PyBullet.
        p.connect(p.GUI if disp else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        assets_path = os.path.dirname(os.path.abspath(__file__))
        p.setAdditionalSearchPath(assets_path)

        # Check PyBullet version (see also the cloth/bag task scripts!).
        p_version = pkg_resources.get_distribution('pybullet').version
        tested = ['2.8.4', '3.0.4', '3.2.6']
        assert p_version in tested, f'PyBullet version {p_version} not in {tested}'

        # Move the camera a little closer to the scene. Most args are not used.
        # PyBullet defaults: yaw=50 and pitch=-35.
        if disp:
            _, _, _, _, _, _, _, _, _, _, _, target = p.getDebugVisualizerCamera()
            p.resetDebugVisualizerCamera(
                cameraDistance=0.9,
                cameraYaw=90,
                cameraPitch=-30,
                cameraTargetPosition=target,)

        # Control PyBullet simulation steps.
        self.step_thread = threading.Thread(target=self.step_simulation)
        self.step_thread.daemon = True
        self.step_thread.start()


    def step_simulation(self):
        """Adding optional hertz parameter for better cloth physics.

        From our discussion with Erwin, we should just set time.sleep(0.001),
        or even consider removing it all together. It's mainly for us to
        visualize PyBullet with the GUI to make it not move too fast
        """
        p.setTimeStep(1.0 / self.hz)
        while True:
            if self.running:
                p.stepSimulation()
            if self.ee is not None:
                self.ee.step()
            time.sleep(0.001)

    def stop(self):
        p.disconnect()
        del self.step_thread

    def start(self):
        self.running = True

    def pause(self):
        self.running = False


    def add_object(self, urdf, pose, category='rigid'):
        """List of (fixed, rigid, or deformable) objects in env."""
        fixed_base = 1 if category == 'fixed' else 0
        obj_id = utils.load_urdf(
            p,
            os.path.join(self.assets_root, urdf),
            pose[0],
            pose[1],
            useFixedBase=fixed_base)
        self.obj_ids[category].append(obj_id)
        return obj_id
    
    def info(self):
        info = {}  # object id : (position, rotation, dimensions)
        for obj_ids in self.obj_ids.values():
            for obj_id in obj_ids:
                pos, rot = p.getBasePositionAndOrientation(obj_id)
                dim = p.getVisualShapeData(obj_id)[0][3]
                info[obj_id] = (pos, rot, dim)
        return info


    #-------------------------------------------------------------------------
    # Standard RL Functions
    #-------------------------------------------------------------------------

    # def step_(self, pick_point, place_point):
    #     _,depth,_ = self.render(self.camera_config_up)

    #     pick_point_tf = utils.pixel_to_position(pick_point,depth[pick_point[1]][pick_point[0]],self.camera_config_up,self.pixel_size)
    #     place_point_tf = utils.pixel_to_position(place_point,depth[place_point[1]][place_point[0]],self.camera_config_up,self.pixel_size)
    #     pick_point_add_ori = [pick_point_tf,[0,0,0,1]]
    #     place_point_add_ori = [place_point_tf,[0,0,0,1]]
    #     #print("pick_point:",pick_point_add_ori)
    #     #print("place_point:",place_point_add_ori)
    #     #print("...............")
    #     self.pick_place(pick_point_add_ori,place_point_add_ori) 
    #     return pick_point_tf,place_point_tf

    def step(self, pick_point, place_point):
        print("(normalized) Pick: ", pick_point, ", Place: ", place_point)
        _,depth,_ = self.render(self.camera_config_up)
        # Convert normalized position to pixel points
        height, width = depth.shape
        pick_point_x, pick_point_y = pick_point
        place_point_x, place_point_y = place_point
        pick_point_x, pick_point_y = int(pick_point_x * width), int(pick_point_y * height)
        place_point_x, place_point_y = int(place_point_x * width), int(place_point_y * height)

        print(f"(pixel) Pick : ({pick_point_x}, {pick_point_y}), Place: ({place_point_x}, {place_point_y})")


        pick_point_tf = utils.pixel_to_position(
            (pick_point_x, pick_point_y),
            depth[pick_point_y][pick_point_x],
            self.camera_config_up,
            self.pixel_size
        )
        place_point_tf = utils.pixel_to_position(
            (place_point_x, place_point_y),
            depth[place_point_y][place_point_x],
            self.camera_config_up,
            self.pixel_size
        )
        pick_point_add_ori = [pick_point_tf,[0,0,0,1]]
        place_point_add_ori = [place_point_tf,[0,0,0,1]]
        #print("pick_point:",pick_point_add_ori)
        #print("place_point:",place_point_add_ori)
        #print("...............")
        self.pick_place(pick_point_add_ori,place_point_add_ori) 
        return pick_point_tf,place_point_tf
        


    def reset(self, disable_render_load=True):
        """Sets up PyBullet, loads models

        Args:
            disable_render_load: Need this as True to avoid `p.loadURDF`
                becoming a time bottleneck, judging from my profiling.
        """
        self.pause()

        self.primitive_params = {
                'speed': 0.01,
                'delta_z': -0.0010,
                'postpick_z': 0.32,
                'preplace_z': 0.32,
                'pause_place': 0.0,      
        }

        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
        self.assets_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)   
        p.setGravity(0, 0, -9.8)

        # Empirically, this seems to make loading URDFs faster w/remote displays.
        if disable_render_load:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        id_plane = p.loadURDF(f'plane/plane.urdf', [0, 0, -0.001])
        id_ws = p.loadURDF(f'table/table.urdf', [0.5, 0, 0])
        #id_ws = p.loadURDF(f'ur5/workspace.urdf', [0.5, 0, 0])

        # Load UR5 robot arm equipped with task-specific end effector.
        self.ur5 = p.loadURDF(f'ur5/ur5-suction.urdf')
        self.ee_tip_link = 12
        self.ee = Suction(self.ur5, 11, self.obj_ids)

        # self.ee = self.task.ee(self.ur5, 9, self.obj_ids)
        # self.ee_tip = 10 

        # Get revolute joint indices of robot (skip fixed joints).
        num_joints = p.getNumJoints(self.ur5)
        joints = [p.getJointInfo(self.ur5, i) for i in range(num_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

        # Move robot to home joint configuration.
        for i in range(len(self.joints)):
            p.resetJointState(self.ur5, self.joints[i], self.homej[i])

        # Get end effector tip pose in home configuration.
        ee_tip_state = p.getLinkState(self.ur5, self.ee_tip_link)
        self.home_pose = np.array(ee_tip_state[0] + ee_tip_state[1])

        # Reset end effector.
        self.ee.release()

        # Setting for deformable object
        #self.ee.set_def_threshold(threshold=self.task.def_threshold)
        #self.ee.set_def_nb_anchors(nb_anchors=self.task.def_nb_anchors)
        assert self.hz >= 480, f'Error, hz={self.hz} is too small!'

        # Restart simulation.
        self.start()
        if disable_render_load:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def render(self,config):
        """Render RGB-D image with specified configuration."""
        # Compute OpenGL camera settings.
        lookdir = np.array([0, 0, 1]).reshape(3, 1)
        updir = np.array([0, -1, 0]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config['rotation'])
        rotm = np.array(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config['position'] + lookdir
        focal_length = config['intrinsics'][0]
        znear, zfar = config['zrange']
        viewm = p.computeViewMatrix(config['position'], lookat, updir)
        fovh = (np.arctan((config['image_size'][0] /
                           2) / focal_length) * 2 / np.pi) * 180
        #self.pixel_size=  0.5 * (config['position'][2] / np.tan(fovh * np.pi / 360)) /config['image_size'][0]

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = config['image_size'][1] / config['image_size'][0]
        projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = p.getCameraImage(
            width=config['image_size'][1],
            height=config['image_size'][0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=1,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        # Get color image.
        color_image_size = (config['image_size'][0],
                            config['image_size'][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        color_image_size = (color_image_size[0], color_image_size[1], 3)
        if config['noise']:
            color = np.int32(color)
            color += np.int32(np.random.normal(0, 3, color_image_size))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (config['image_size'][0], config['image_size'][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
        depth = (2. * znear * zfar) / depth
        if config['noise']:
            depth += np.random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)


        return color, depth, segm

    #-------------------------------------------------------------------------
    # Record Functions
    #-------------------------------------------------------------------------

    def record_video(self,file_name,save_dir):

        dir_name = os.path.join('record/',save_dir)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        self.curr_recording= p.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4,
                os.path.join(dir_name, '{}.mp4'.format(file_name)))
    
    def stop_record(self):
         p.stopStateLogging(self.curr_recording)


    #-------------------------------------------------------------------------
    # Robot Movement Functions
    #-------------------------------------------------------------------------

    def movej(self, targj, speed=0.01, t_lim=20):
        """Move UR5 to target joint configuration."""
        t0 = time.time()
        while (time.time() - t0) < t_lim:
            currj = [p.getJointState(self.ur5, i)[0] for i in self.joints]
            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return True

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            p.setJointMotorControlArray(
                bodyIndex=self.ur5,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)
            time.sleep(0.001)
        print('Warning: movej exceeded {} sec timeout. Skipping.'.format(t_lim))
        return False

    def movep(self, pose, speed=0.01):
        """Move UR5 to target end effector pose."""
        # # Keep joint angles between -180/+180
        # targj[5] = ((targj[5] + np.pi) % (2 * np.pi) - np.pi)
        targj = self.solve_IK(pose)
        return self.movej(targj, speed, self.t_lim)

    def solve_IK(self, pose):
        homej_list = np.array(self.homej).tolist()
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5,
            endEffectorLinkIndex=self.ee_tip_link,
            targetPosition=pose[:3],
            targetOrientation=pose[3:],
            lowerLimits=[-17, -2.3562, -17, -17, -17, -17],
            upperLimits=[17, 0, 17, 17, 17, 17],
            jointRanges=[17] * 6,
            restPoses=homej_list,
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.array(joints)
        joints[joints > 2 * np.pi] = joints[joints > 2 * np.pi] - 2 * np.pi
        joints[joints < -2 * np.pi] = joints[joints < -2 * np.pi] + 2 * np.pi
        return joints

    #-------------------------------------------------------------------------
    # Motion Primitives
    #-------------------------------------------------------------------------

    def pick_place(self, pose0, pose1):
        """Execute pick and place primitive.
            pose0: picking pose.
            pose1: placing pose.

        Returns:
            A bool indicating whether the action succeeded or not, via
            checking the sequence of movep calls. If any movep failed, then
            self.step() will terminate the episode after this action.
        """
        #initial params
        speed = 0.01
        delta_z = -0.001
        prepick_z = 0.3
        postpick_z = 0.3
        preplace_z = 0.3
        pause_place = 0.0
        final_z = 0.3

        # adjust action params
        speed       = self.primitive_params['speed']
        delta_z     = self.primitive_params['delta_z']
        postpick_z  = self.primitive_params['postpick_z']
        preplace_z  = self.primitive_params['preplace_z']
        pause_place = self.primitive_params['pause_place']

        # Otherwise, proceed as normal.
        success = True
        pick_position = np.array(pose0[0])
        pick_rotation = np.array(pose0[1])
        prepick_position = pick_position.copy()
        prepick_position[2] = prepick_z

        # Execute picking motion primitive.
        prepick_pose = np.hstack((prepick_position, pick_rotation))
        success &= self.movep(prepick_pose)
        target_pose = prepick_pose.copy()
        delta = np.array([0, 0, delta_z, 0, 0, 0, 0])

        # Lower gripper until (a) touch object (rigid OR softbody), or (b) hit ground.
        while not self.ee.detect_contact() and target_pose[2] > 0:
            target_pose += delta
            success &= self.movep(target_pose)

        # Create constraint (rigid objects) or anchor (deformable).
        self.ee.activate()

        # Increase z slightly (or hard-code it) and check picking success.

        prepick_pose[2] = postpick_z
        success &= self.movep(prepick_pose, speed=speed)
        time.sleep(pause_place) # extra rest for bags
 
        pick_success = self.ee.check_grasp()

        if pick_success:
            place_position = np.array(pose1[0])
            place_rotation = np.array(pose1[1])
            preplace_position = place_position.copy()
            preplace_position[2] = 0.3 + pick_position[2]

            # Execute placing motion primitive if pick success.
            preplace_pose = np.hstack((preplace_position, place_rotation))
            preplace_pose[2] = preplace_z
            success &= self.movep(preplace_pose, speed=speed)
            time.sleep(pause_place) # extra rest for bags
            
            target_pose = preplace_pose.copy()
            while not self.ee.detect_contact() and target_pose[2] > 0:
                target_pose += delta
                success &= self.movep(target_pose)

            # Release AND get gripper high up, to clear the view for images.
            self.ee.release()
            preplace_pose[2] = final_z
            success &= self.movep(preplace_pose)
        else:
            # Release AND get gripper high up, to clear the view for images.
            self.ee.release()
            prepick_pose[2] = final_z
            success &= self.movep(prepick_pose)
        # Move robot to home joint configuration.
        initial_pos = np.array([0, 0.487, 0.32,0,0,0,1])
        success_initial = self.movep(initial_pos)
        if not success_initial:
            # Move robot to home joint configuration.
            for i in range(len(self.joints)):
                p.resetJointState(self.ur5, self.joints[i], self.homej[i])

        return success

    def sweep(self, pose0, pose1):
        """Execute sweeping primitive."""
        success = True
        position0 = np.float32(pose0[0])
        position1 = np.float32(pose1[0])
        direction = position1 - position0
        length = np.linalg.norm(position1 - position0)
        if length == 0:
            direction = np.float32([0, 0, 0])
        else:
            direction = (position1 - position0) / length

        theta = np.arctan2(direction[1], direction[0])
        rotation = p.getQuaternionFromEuler((0, 0, theta))

        over0 = position0.copy()
        over0[2] = 0.3
        over1 = position1.copy()
        over1[2] = 0.3

        success &= self.movep(np.hstack((over0, rotation)))
        success &= self.movep(np.hstack((position0, rotation)))

        num_pushes = np.int32(np.floor(length / 0.01))
        for i in range(num_pushes):
            target = position0 + direction * num_pushes * 0.01
            success &= self.movep(np.hstack((target, rotation)), speed=0.003)

        success &= self.movep(np.hstack((position1, rotation)), speed=0.003)
        success &= self.movep(np.hstack((over1, rotation)))
        return success

    def push(self, pose0, pose1):
        """Execute pushing primitive."""
        p0 = np.float32(pose0[0])
        p1 = np.float32(pose1[0])
        p0[2], p1[2] = 0.025, 0.025
        if np.sum(p1 - p0) == 0:
            push_direction = 0
        else:
            push_direction = (p1 - p0) / np.linalg.norm((p1 - p0))
        p1 = p0 + push_direction * 0.01
        success = True
        success &= self.movep(np.hstack((p0, self.home_pose[3:])))
        success &= self.movep(np.hstack((p1, self.home_pose[3:])), speed=0.003)
        return success
