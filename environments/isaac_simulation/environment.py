import os
import sys
import pathlib
from pathlib import Path
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
from PIL import Image


class Environment():
    def __init__(self,asset_root):
        self.asset_root = asset_root
        self.sim = None
        self.env = None
        self.reset()

    def reset(self):
        self.set_sim()
        self.set_ground()
        self.set_env()
        self.add_init_objects()
        self.set_camera()

    def set_sim(self):
        gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.use_gpu = True
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        sim_params.dt = 1.0 / 60.0
        # sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

        # create sim
        self.sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    def set_ground(self):
        gym = gymapi.acquire_gym()
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
        # plane_params.distance = 0
        # plane_params.static_friction = 1
        # plane_params.dynamic_friction = 1
        # plane_params.restitution = 0

        # create the ground plane
        gym.add_ground(self.sim, plane_params)

    def set_env(self):
        gym = gymapi.acquire_gym()
        num_envs = 1
        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.env = gym.create_env(self.sim, env_lower, env_upper, num_envs)

    def add_init_objects(self):
        gym = gymapi.acquire_gym()
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        #asset_options.armature = 0.01
        asset_options.use_mesh_materials = True
        asset_options.armature = 0.001
        #asset_options.fix_base_link = True
        asset_options.thickness = 0.002
        #asset_options.disable_gravity = 0
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

        table_asset = gym.load_asset(self.sim, self.asset_root, "table/table.urdf", asset_options)
        
        asset_options.fix_base_link = False # allow the laptop to fall
        
        laptop_asset = gym.load_asset(self.sim, self.asset_root, "laptop/laptop.urdf", asset_options)
        cup_asset = gym.load_asset(self.sim, self.asset_root, "yellow_cup/model.urdf", asset_options)

        pose_table = gymapi.Transform()

        pose_table.p = gymapi.Vec3(0, 3., 0.02)
        pose_table.r = gymapi.Quat(0, 0, 0, 1)
        #pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), -0.5 * math.pi)
        table_handle = gym.create_actor(self.env,table_asset,pose_table,"table",0,0)

        

        pose_labtop = gymapi.Transform()
        pose_labtop.p = gymapi.Vec3(-0.1, 3.3, 0.2)
        pose_labtop.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0.09, 0.09, 2), 0.5 * np.pi)
        labtop_handle = gym.create_actor(self.env,laptop_asset,pose_labtop,"laptop",0,0)

        pose_cup = gymapi.Transform()
        pose_cup.p = gymapi.Vec3(0.1, 2.8, 1)# higher than the table
        pose_cup.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0., 0, 1), 0.5 * np.pi)
        cup_handle = gym.create_actor(self.env,cup_asset,pose_cup,"cup",0,0)

        box_size = 0.05
        box_asset = gym.create_box(self.sim, box_size, box_size, box_size, asset_options)
        pose_box = gymapi.Transform()
        pose_box.p = gymapi.Vec3(0.1, 3, 0.4)
        pose_box.r = gymapi.Quat(0, 0, 0, 1)
        box_handle = gym.create_actor(self.env, box_asset, pose_box, "box", 0, 0)



    def set_camera(self):
        gym = gymapi.acquire_gym()
        camera_property = gymapi.CameraProperties()
        camera_property.width = 1920
        camera_property.height = 1080

        camera_handle = gym.create_camera_sensor(self.env,camera_property)
        # camera_position = gymapi.Vec3(0, 3, 2)
        # camera_target = gymapi.Vec3(0.3, 3, 0.)
        # gym.set_camera_location(camera_handle,env, camera_position, camera_target)
        camera_transform = gymapi.Transform()
        camera_transform.p = gymapi.Vec3(0, 3, 0.61)
        camera_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.math.pi/2) 

        # camera_transform.p = gymapi.Vec3(-1, 3, 0.05)
        # camera_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), 0.) 


        gym.set_camera_transform(camera_handle,self.env,camera_transform)
        self.camera_properties = camera_property


    def loop_test(self):
        gym = gymapi.acquire_gym()
        viewer = gym.create_viewer(self.sim, self.camera_properties)
        gym.subscribe_viewer_keyboard_event(viewer,gymapi.KEY_R,"reset")
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(-1, 3, 0.05), gymapi.Vec3(0, 3, 0))

        while not gym.query_viewer_has_closed(viewer):
            gym.simulate(self.sim)
            gym.fetch_results(self.sim, True)

            # update the viewer
            gym.step_graphics(self.sim)
            gym.draw_viewer(viewer, self.sim, True)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            gym.sync_frame_time(self.sim) 

            # if gym.query_viewer_key_press(viewer,gymapi.KEY_R):
            #     print("reset")
            #     gym.reset_sim(sim)

        gym.destroy_viewer(viewer)
        gym.destroy_sim(self.sim)


if __name__ == "__main__":
    asset_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets")
    env = Environment(asset_root)
    env.loop_test()