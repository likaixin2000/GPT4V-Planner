import os
import sys
import pathlib
from pathlib import Path
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from PIL import Image
# from environments.isaac_simulation.examples.util import *
import torch 


CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


class Environment():
    def __init__(self,asset_root):
        self.asset_root = asset_root
        self.sim = None
        self.env = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.handle_map = {}
        self.camera_handle = None

    def reset(self):
        self.set_sim()
        self.set_ground()
        self.set_env()
        self.set_table()

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

    def set_table(self):
        gym = gymapi.acquire_gym()
        self.table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
        assert_options = gymapi.AssetOptions()
        assert_options.fix_base_link = True
        table_asset = gym.create_box(self.sim, self.table_dims.x, self.table_dims.y, self.table_dims.z, assert_options)
        self.table_pose = gymapi.Transform()
        self.table_pose.p = gymapi.Vec3(0.5, 0, 0.5*self.table_dims.z)
        # table_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.table_handle = gym.create_actor(self.env, table_asset, self.table_pose, "table", 0, 0)
        # 设置一个浅灰色的颜色
        color = gymapi.Vec3(0.7, 0.7, 0.7)
        gym.set_rigid_body_color(self.env, self.table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    def set_env(self):
        gym = gymapi.acquire_gym()
        num_envs = 1
        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.env = gym.create_env(self.sim, env_lower, env_upper, num_envs)

    def empty_step(self,n_steps):
        gym = gymapi.acquire_gym()
        for i in range(n_steps):
            gym.simulate(self.sim)
            gym.fetch_results(self.sim, True)
            gym.step_graphics(self.sim)

    def set_camera_by_target(self,position,target):
        gym = gymapi.acquire_gym()
        camera_property = gymapi.CameraProperties()
        camera_property.width = CAMERA_WIDTH
        camera_property.height = CAMERA_HEIGHT
        camera_handle = gym.create_camera_sensor(self.env,camera_property)
        gym.set_camera_location(camera_handle,self.env,position,target)
        self.camera_properties = camera_property
        self.camera_handle = camera_handle
        self.camera_pose = position

    def set_camera_by_transform(self,transform):
        gym = gymapi.acquire_gym()
        camera_property = gymapi.CameraProperties()
        camera_property.width = CAMERA_WIDTH
        camera_property.height = CAMERA_HEIGHT
        camera_handle = gym.create_camera_sensor(self.env,camera_property)
        gym.set_camera_transform(camera_handle,self.env,transform)
        self.camera_properties = camera_property
        self.camera_handle = camera_handle
        self.camera_pose = transform.p


    def set_look_down_camera(self):
        gym = gymapi.acquire_gym()
        # camera_position = gymapi.Vec3(self.table_pose.p.x-0.01, self.table_pose.p.y, self.table_pose.p.z + 0.8)
        # camera_target = gymapi.Vec3(self.table_pose.p.x, self.table_pose.p.y, self.table_pose.p.z)
        # self.set_camera_by_target(camera_position,camera_target)

        ## 摄像机默认朝着x 朝下看需要沿着y轴
        camera_transform = gymapi.Transform()
        camera_transform.p = gymapi.Vec3(self.table_pose.p.x, self.table_pose.p.y, self.table_dims.z+0.6)
        camera_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.math.pi/2)
        self.set_camera_by_transform(camera_transform) 

    

    def set_look_ahead_camera(self):
        gym = gymapi.acquire_gym()
        
        # 侧视
        camera_transform = gymapi.Transform()
        camera_transform.p = gymapi.Vec3(self.table_pose.p.x-self.table_dims.x, self.table_pose.p.y, 0.5 )
        # camera_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0) 
        self.set_camera_by_transform(camera_transform)

    def add_object_relative_to_table(self,object_name,urdf_path,pose,axis_angle,fix_base_link=False):
        gym = gymapi.acquire_gym()
        asset_options = gymapi.AssetOptions()
        #asset_options.armature = 0.01
        asset_options.use_mesh_materials = True
        asset_options.armature = 0.001
        #asset_options.fix_base_link = True
        asset_options.thickness = 0.002
        #asset_options.disable_gravity = 0
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    
        asset_options.fix_base_link = fix_base_link # allow the laptop to fall

        asset = gym.load_asset(self.sim, self.asset_root, urdf_path, asset_options)
        asset_pose = gymapi.Transform()
        asset_pose.p = gymapi.Vec3(self.table_pose.p.x + pose[0], self.table_pose.p.y + pose[1], self.table_dims.z + pose[2])
        asset_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(axis_angle[0], axis_angle[1], axis_angle[2]), axis_angle[3])
        handle = gym.create_actor(self.env,asset,asset_pose,object_name,0,0)
        self.handle_map[object_name] = handle

    def add_box_relative_to_table(self,box_name,box_dims,pose,axis_angle,color=None,fix_base_link=False):
        gym = gymapi.acquire_gym()
        asset_options = gymapi.AssetOptions()
        asset_options.use_mesh_materials = True
        asset_options.armature = 0.001
        asset_options.thickness = 0.002
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.fix_base_link = fix_base_link

        asset = gym.create_box(self.sim, box_dims[0], box_dims[1], box_dims[2], asset_options)
        asset_pose = gymapi.Transform()
        asset_pose.p = gymapi.Vec3(self.table_pose.p.x + pose[0], self.table_pose.p.y + pose[1], self.table_dims.z + pose[2])
        asset_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(axis_angle[0], axis_angle[1], axis_angle[2]), axis_angle[3])
        handle = gym.create_actor(self.env,asset,asset_pose,box_name,0,0)
        self.handle_map[box_name] = handle
        if color is not None:
            vcolor = gymapi.Vec3(color[0], color[1], color[2])
            gym.set_rigid_body_color(self.env, handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, vcolor)
        else:
            vcolor = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            gym.set_rigid_body_color(self.env, handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # return RGBA
    def render(self) -> np.ndarray:
        if self.camera_handle is None:
            self.set_look_down_camera()
        gym = gymapi.acquire_gym()
        gym.render_all_camera_sensors(self.sim)
        image_np = gym.get_camera_image(self.sim,self.env,self.camera_handle,gymapi.IMAGE_COLOR)
        image_np= image_np.reshape((self.camera_properties.height,self.camera_properties.width,4))
        return image_np
    
    def get_depth(self):
        if self.camera_handle is None:
            self.set_look_down_camera()
        gym = gymapi.acquire_gym()
        gym.render_all_camera_sensors(self.sim)
        depth_image = gym.get_camera_image(self.sim,self.env, self.camera_handle, gymapi.IMAGE_DEPTH)

        depth_image[depth_image == -np.inf] = 0

                # clamp depth image to 10 meters to make output image human friendly
        depth_image[depth_image < -10] = -10

        return depth_image
    
    def render_depth(self):
        depth_image = self.get_depth()
        normalized_depth = -255.0*(depth_image/np.min(depth_image + 1e-4))
        normalized_depth_image = Image.fromarray(normalized_depth.astype(np.uint8), mode="L")
        return normalized_depth_image
    
    def pick_place(self,pick_point,place_point):
        gym = gymapi.acquire_gym()
        pick_x, pick_y ,pick_z = pick_point
        place_x, place_y, place_z = place_point
        dists=[]
        for name,handle in self.handle_map.items():
            body_pose = gym.get_actor_rigid_body_states(self.env,handle,gymapi.STATE_POS)
            dists.append((np.linalg.norm([body_pose['pose']['p']['x']-pick_x,body_pose['pose']['p']['y']-pick_y]),name,handle))
        dists.sort(key=lambda x:x[0])
        if dists[0][0]>0.1:
            print("no object close to pick point")
            return None
        else:
            print("pick object distance",dists[0][1])
        pick_name = dists[0][1]
        pick_handle = dists[0][2]
        print("pick_name",pick_name)
        print("pick_handle",pick_handle)
        body_pose = gym.get_actor_rigid_body_states(self.env,pick_handle,gymapi.STATE_POS)
        body_pose['pose']['p']['x'] = place_x
        body_pose['pose']['p']['y'] = place_y
        # body_pose['pose']['p']['z'] = self.table_dims.z + 0.08
        gym.set_actor_rigid_body_states(self.env,pick_handle,body_pose,gymapi.STATE_POS)
        self.empty_step(1)
        gym.render_all_camera_sensors(self.sim)
        return pick_name
            
        
    # example
    def add_init_objects(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        gym = gymapi.acquire_gym()
        box_size = 0.05
        box_asset = gym.create_box(self.sim, box_size, box_size, box_size, asset_options)
        pose_box = gymapi.Transform()
        #pose_box.p = gymapi.Vec3(0.1, 3, 0.4)
        pose_box.p = gymapi.Vec3(self.table_pose.p.x , self.table_pose.p.y, self.table_pose.p.z + 0.3)
        pose_box.r = gymapi.Quat(0, 0, 0, 1)
        box_handle = gym.create_actor(self.env, box_asset, pose_box, "box2", 0, 0)
        color = gymapi.Vec3(0, 0.7, 0.7)
        gym.set_rigid_body_color(self.env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # self.handle_map['box']=box_handle
        self.add_object_relative_to_table("laptop","laptop/laptop.urdf",[-0.1, 0.3, 0.2],[0.09, 0.09, 2, 0.5 * np.pi])
        self.add_object_relative_to_table("cup","yellow_cup/model.urdf",[0.1, -0.2, 0.3],[0., 0, 1, 0.5 * np.pi])
        self.add_box_relative_to_table("box",[0.05, 0.05, 0.05],[0, -0.1, 0.1],[0, 0, 1, 0],[0, 0.7, 0.7])



    def loop_test(self):
        gym = gymapi.acquire_gym()
        viewer = gym.create_viewer(self.sim, self.camera_properties)
        gym.subscribe_viewer_keyboard_event(viewer,gymapi.KEY_R,"reset")
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(-0.01, 3, 1.1), gymapi.Vec3(0, 3, 0))

        while not gym.query_viewer_has_closed(viewer):
            gym.simulate(self.sim)
            gym.fetch_results(self.sim, True)

            # update the viewer
            gym.step_graphics(self.sim)
            gym.draw_viewer(viewer, self.sim, True)
            gym.sync_frame_time(self.sim) 

        gym.destroy_viewer(viewer)
        gym.destroy_sim(self.sim)

    # no viewer
    def camera_test(self):
        gym = gymapi.acquire_gym()
        self.add_init_objects()
        self.set_look_down_camera()
        self.empty_step(60)
        gym.render_all_camera_sensors(self.sim)
        image_np = gym.get_camera_image(self.sim,self.env,self.camera_handle,gymapi.IMAGE_COLOR)
        image_np= image_np.reshape((self.camera_properties.height,self.camera_properties.width,4))
        print(image_np.shape)
        image_np = image_np[:,:,:3]
        image = Image.fromarray(image_np ,mode="RGB")
        image.save("camera_test1.png")
        #gym.write_camera_image_to_file(self.sim,self.env,self.camera_handle,gymapi.IMAGE_COLOR,"camera_test1.png")

        ## try to move the box
        box_handle = self.handle_map['box']
        body_pose = gym.get_actor_rigid_body_states(self.env,box_handle,gymapi.STATE_POS)
        print("body_pose",body_pose)
        body_pose['pose']['p']['x'] -= 0.2
        gym.set_actor_rigid_body_states(self.env,box_handle,body_pose,gymapi.STATE_POS)
        self.empty_step(60)
        gym.render_all_camera_sensors(self.sim)
        image_np = gym.get_camera_image(self.sim,self.env,self.camera_handle,gymapi.IMAGE_COLOR)
        image_np= image_np.reshape((self.camera_properties.height,self.camera_properties.width,4))
        print(image_np.shape)
        image = Image.fromarray(image_np ,mode="RGBA")
        image.save("camera_test2.png")

        depth =self.get_depth()
        print("depth shape",depth.shape)
        pred = depth[360][900]
        for idx,depth_value in enumerate(depth[360][900:]):
            if np.abs(depth_value - pred)>0.2:
                print("depth value change high dix:",idx + 900)
                print(f"pred:{pred},current:{depth_value}")
                break
            pred = depth_value

        depth_image = self.render_depth()
        print(depth_image.size)
        depth_image.save("depth_test.png")


        gym.destroy_sim(self.sim)



if __name__ == "__main__":
    asset_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "assets")
    myenv = Environment(asset_root)
    myenv.reset()
    myenv.camera_test()