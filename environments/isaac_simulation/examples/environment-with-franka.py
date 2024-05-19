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
from util import *
import torch 

class Environment():
    def __init__(self,asset_root):
        self.asset_root = asset_root
        self.sim = None
        self.env = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.reset()

    def reset(self):
        self.set_sim()
        self.set_ground()
        self.set_env()
        self.add_init_objects()
        self.set_camera()
        self.franka_info=self.set_franka()

    def set_sim(self):
        gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.001
        sim_params.physx.use_gpu = True
        sim_params.physx.contact_offset = 0.001
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.rest_offset = 0.0
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
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

        pose_table.p = gymapi.Vec3(0, 3., 0.02+0.45)
        pose_table.r = gymapi.Quat(0, 0, 0, 1)
        #pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), -0.5 * math.pi)
        table_handle = gym.create_actor(self.env,table_asset,pose_table,"table",0,0)

        

        pose_labtop = gymapi.Transform()
        pose_labtop.p = gymapi.Vec3(-0.1, 3.3, 0.2+0.7)
        pose_labtop.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0.09, 0.09, 2), 0.5 * np.pi)
        labtop_handle = gym.create_actor(self.env,laptop_asset,pose_labtop,"laptop",0,0)

        pose_cup = gymapi.Transform()
        pose_cup.p = gymapi.Vec3(0.1, 2.8, 0.5+0.7)# higher than the table
        pose_cup.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0., 0, 1), 0.5 * np.pi)
        cup_handle = gym.create_actor(self.env,cup_asset,pose_cup,"cup",0,0)
        self.cup_idxs=[gym.get_actor_rigid_body_index(self.env, cup_handle, 0, gymapi.DOMAIN_SIM)]

        box_size = 0.045
        asset_options = gymapi.AssetOptions()
        box_asset = gym.create_box(self.sim, box_size, box_size, box_size, asset_options)
        pose_box = gymapi.Transform()
        pose_box.p = gymapi.Vec3(0.1, 3, 0.2+0.7)
        pose_box.r = gymapi.Quat(0, 0, 0, 1)
        box_handle = gym.create_actor(self.env, box_asset, pose_box, "box", 0, 0)

        self.box_idxs=[gym.get_actor_rigid_body_index(self.env, box_handle, 0, gymapi.DOMAIN_SIM)]

        n_steps = 60
        for i in range(n_steps):
            gym.simulate(self.sim)
            gym.fetch_results(self.sim, True)
            gym.step_graphics(self.sim)
            # gym.step_graphics(self.sim)



    def set_camera(self):
        gym = gymapi.acquire_gym()
        camera_property = gymapi.CameraProperties()
        camera_property.width = 1920
        camera_property.height = 1080

        camera_handle = gym.create_camera_sensor(self.env,camera_property)

        # 俯视
        # camera_position = gymapi.Vec3(0, 3, 3)
        # camera_target = gymapi.Vec3(0.3, 3, 0.)
        # gym.set_camera_location(camera_handle,self.env, camera_position, camera_target)
        
        # 侧视
        camera_transform = gymapi.Transform()
        camera_transform.p = gymapi.Vec3(0, 3, 0.61)
        camera_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.math.pi/2) 
        gym.set_camera_transform(camera_handle,self.env,camera_transform)
        
        self.camera_properties = camera_property

    def set_franka(self):
        # load franka asset
        gym = gymapi.acquire_gym()
        num_envs = 1
        controller = "ik" ## or "osc"
        # Set controller parameters
        # IK params
        damping = 0.05

        # OSC params
        kp = 150.
        kd = 2.0 * np.sqrt(kp)
        kp_null = 10.
        kd_null = 2.0 * np.sqrt(kp_null)

        franka_asset_file = "franka_description/robots/franka_panda.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        franka_asset = gym.load_asset(self.sim, self.asset_root, franka_asset_file, asset_options)

        # configure franka dofs
        franka_dof_props = gym.get_asset_dof_properties(franka_asset)
        franka_lower_limits = franka_dof_props["lower"]
        franka_upper_limits = franka_dof_props["upper"]
        franka_ranges = franka_upper_limits - franka_lower_limits
        franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

        # use position drive for all dofs
        if controller == "ik":
            franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
            franka_dof_props["stiffness"][:7].fill(400.0)
            franka_dof_props["damping"][:7].fill(40.0)
        else:       # osc
            franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
            franka_dof_props["stiffness"][:7].fill(0.0)
            franka_dof_props["damping"][:7].fill(0.0)
        # grippers
        franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][7:].fill(800.0)
        franka_dof_props["damping"][7:].fill(40.0)

        # default dof states and position targets
        franka_num_dofs = gym.get_asset_dof_count(franka_asset)
        default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
        default_dof_pos[:7] = franka_mids[:7]
        # grippers open
        default_dof_pos[7:] = franka_upper_limits[7:]

        default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos

        # # send to torch (osc only)
        # default_dof_pos_tensor = to_torch(default_dof_pos, device=self.device)

        # get link index of panda hand, which we will use as end effector
        franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)
        franka_hand_index = franka_link_dict["panda_hand"]

        franka_pose = gymapi.Transform()
        franka_pose.p = gymapi.Vec3(0.5,3, 0.25)
        #绕z轴旋转180度
        franka_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.pi)

        # add franka to the scene
        franka_handle = gym.create_actor(self.env, franka_asset, franka_pose, "franka", 0, 0) # last para maybe 2

        #
        gym.set_actor_dof_properties(self.env, franka_handle, franka_dof_props)
        gym.set_actor_dof_states(self.env, franka_handle, default_dof_state,gymapi.STATE_ALL)
        # set initial position targets
        gym.set_actor_dof_position_targets(self.env, franka_handle, default_dof_pos)

        # get inital hand pose
        hand_handle = gym.find_actor_rigid_body_handle(self.env, franka_handle, "panda_hand")
        hand_pose = gym.get_rigid_transform(self.env, hand_handle)


        # init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
        # init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])
        # get global index of hand in rigid body state tensor
        hand_idx = gym.find_actor_rigid_body_index(self.env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
        # hand_idx=0
        self.hand_idxs = [hand_idx]


        ## actions and poses

        gym.prepare_sim(self.sim)

        init_pos = torch.Tensor([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z]).view(1,3).to(self.device)
        init_rot = torch.Tensor([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w]).view(1,4).to(self.device)

        # hand orientation for grasping
        # down_q = torch.stack(num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).to(device).view((num_envs, 4))

        # box corner coords, used to determine grasping yaw
        box_size = 0.05
        box_half_size = 0.5 * box_size
        corner_coord = torch.Tensor([box_half_size, box_half_size, box_half_size])
        corners = torch.stack(num_envs * [corner_coord]).to(device)

        # downard axis
        # down_dir = torch.Tensor([0, 0, -1]).to(device).view(1, 3)

        # get jacobian tensor
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        _jacobian = gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to franka hand
        # except for franka hand
        j_eef = jacobian[:, franka_hand_index - 1, :, :7].to(self.device)

        # get mass matrix tensor
        _massmatrix = gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        mm = mm[:, :7, :7]          # only need elements corresponding to the franka arm

        # get rigid body state tensor
        _rb_states = gym.acquire_rigid_body_state_tensor(self.sim)
        rb_states = gymtorch.wrap_tensor(_rb_states)

        # get dof state tensor
        _dof_states = gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states).to(self.device)
        dof_pos = dof_states[:, 0].view(num_envs, 9, 1)
        dof_vel = dof_states[:, 1].view(num_envs, 9, 1)

        # Create a tensor noting whether the hand should return to the initial position
        hand_restart = torch.full([num_envs], False, dtype=torch.bool).to(device)

        # Set action tensors
        pos_action = torch.zeros_like(dof_pos).squeeze(-1).to(self.device)
        effort_action = torch.zeros_like(pos_action)

        def control_ik(dpose):
            # solve damped least squares
            j_eef_T = torch.transpose(j_eef, 1, 2).to(self.device)
            lmbda = torch.eye(6, device=self.device) * (damping ** 2)
            u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)
            return u

        return {
            'controller': controller,
            'box_size': box_size,
            'rb_states': rb_states,
            'corners': corners,
            'init_pos': init_pos,
            'init_rot': init_rot,
            'pose_act': pos_action,
            'effort_act': effort_action,
            'dof_pos': dof_pos,
            'control_ik': control_ik,
            'hand_restart': hand_restart
        }


    def loop_test(self,info):
        gym = gymapi.acquire_gym()
        viewer = gym.create_viewer(self.sim, self.camera_properties)
        gym.subscribe_viewer_keyboard_event(viewer,gymapi.KEY_R,"reset")
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1, 3, 0.9), gymapi.Vec3(0, 3, 0.7))
        
        ## init information from franka_info init
        num_envs=1
        controller=info["controller"]
        rb_states=info["rb_states"]
        box_size=info["box_size"]
        corners=info["corners"]
        init_pos=info["init_pos"]
        init_rot=info["init_rot"]
        pos_action=info["pose_act"]
        effort_action=info["effort_act"]
        dof_pos=info["dof_pos"]
        control_ik=info["control_ik"]
        hand_restart=info["hand_restart"]



        while not gym.query_viewer_has_closed(viewer):
            gym.simulate(self.sim)
            gym.fetch_results(self.sim, True)

    # refresh tensors
            gym.refresh_rigid_body_state_tensor(self.sim)
            gym.refresh_dof_state_tensor(self.sim)
            gym.refresh_jacobian_tensors(self.sim)
            gym.refresh_mass_matrix_tensors(self.sim)

            # box_pos = rb_states[self.box_idxs, :3].to(self.device)
            # box_rot = rb_states[self.box_idxs, 3:7].to(self.device)
            box_pos = rb_states[self.cup_idxs, :3].to(self.device)
            box_rot = rb_states[self.cup_idxs, 3:7].to(self.device)

            hand_pos = rb_states[self.hand_idxs, :3].to(self.device)
            hand_rot = rb_states[self.hand_idxs, 3:7].to(self.device)
            hand_vel = rb_states[self.hand_idxs, 7:].to(self.device)

            # downard axis
            down_dir = torch.Tensor([0, 0, -1]).to(self.device).view(1, 3)   
            to_box = box_pos - hand_pos
            box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1).to(self.device)
            box_dir = to_box / box_dist
            box_dot = box_dir @ down_dir.view(3, 1)

            # how far the hand should be from box for grasping
            grasp_offset = 0.11 if controller == "ik" else 0.10

            # determine if we're holding the box (grippers are closed and box is near)
            gripper_sep = dof_pos[:, 7] + dof_pos[:, 8]
            gripped = (gripper_sep < 0.045) & (box_dist < grasp_offset + 0.5 * box_size)

            yaw_q = cube_grasping_yaw(box_rot, corners)
            box_yaw_dir = quat_axis(yaw_q, 0)
            hand_yaw_dir = quat_axis(hand_rot, 0)
            yaw_dot = torch.bmm(box_yaw_dir.view(num_envs, 1, 3), hand_yaw_dir.view(num_envs, 3, 1)).squeeze(-1)

            # determine if we have reached the initial position; if so allow the hand to start moving to the box
            to_init = init_pos - hand_pos
            init_dist = torch.norm(to_init, dim=-1)
            hand_restart = (hand_restart & (init_dist > 0.02)).squeeze(-1)
            return_to_start = (hand_restart | gripped.squeeze(-1)).unsqueeze(-1)

            # if hand is above box, descend to grasp offset
            # otherwise, seek a position above the box
            above_box = ((box_dot >= 0.99) & (yaw_dot >= 0.95) & (box_dist < grasp_offset * 3)).squeeze(-1)
            grasp_pos = box_pos.clone()
            grasp_pos[:, 2] = torch.where(above_box, box_pos[:, 2] + grasp_offset, box_pos[:, 2] + grasp_offset * 2.5)

            # compute goal position and orientation
            down_q = torch.stack(num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).to(device).view((num_envs, 4))
            goal_pos = torch.where(return_to_start, init_pos, grasp_pos)
            goal_rot = torch.where(return_to_start, init_rot, quat_mul(down_q, quat_conjugate(yaw_q)))

            # compute position and orientation error
            pos_err = goal_pos - hand_pos
            orn_err = orientation_error(goal_rot, hand_rot)
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1).to(self.device)

            # Deploy control based on type
            if controller == "ik":
                pos_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + control_ik(dpose)
            else:       # osc
                # effort_action[:, :7] = control_osc(dpose)
                pass

            # gripper actions depend on distance between hand and box
            close_gripper = (box_dist < grasp_offset + 0.02) | gripped
            # always open the gripper above a certain height, dropping the box and restarting from the beginning
            hand_restart = hand_restart | (box_pos[:, 2] > 0.6)
            keep_going = torch.logical_not(hand_restart)
            close_gripper = close_gripper & keep_going.unsqueeze(-1)
            grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * num_envs).to(device), torch.Tensor([[0.04, 0.04]] * num_envs).to(device))
            pos_action[:, 7:9] = grip_acts

            print("posaction:",pos_action)

            # Deploy actions
            pos_action = pos_action.to("cpu")
            effort_action = effort_action.to("cpu")
            gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))
            gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(effort_action))

            # update the viewer
            gym.step_graphics(self.sim)
            gym.draw_viewer(viewer, self.sim, True)
            gym.sync_frame_time(self.sim) 

        gym.destroy_viewer(viewer)
        gym.destroy_sim(self.sim)


if __name__ == "__main__":
    asset_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../assets")
    myenv = Environment(asset_root)
    myenv.loop_test(myenv.franka_info)