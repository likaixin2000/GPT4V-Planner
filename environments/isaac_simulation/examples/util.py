from isaacgym.torch_utils import *
import torch 
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats


# def control_ik(dpose):
#     global damping, j_eef, num_envs
#     # solve damped least squares
#     j_eef_T = torch.transpose(j_eef, 1, 2)
#     lmbda = torch.eye(6, device=device) * (damping ** 2)
#     u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)
#     return u


# def control_osc(dpose):
#     global kp, kd, kp_null, kd_null, default_dof_pos_tensor, mm, j_eef, num_envs, dof_pos, dof_vel, hand_vel
#     mm_inv = torch.inverse(mm)
#     m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
#     m_eef = torch.inverse(m_eef_inv)
#     u = torch.transpose(j_eef, 1, 2) @ m_eef @ (
#         kp * dpose - kd * hand_vel.unsqueeze(-1))

#     # Nullspace control torques `u_null` prevents large changes in joint configuration
#     # They are added into the nullspace of OSC so that the end effector orientation remains constant
#     # roboticsproceedings.org/rss07/p31.pdf
#     j_eef_inv = m_eef @ j_eef @ mm_inv
#     u_null = kd_null * -dof_vel + kp_null * (
#         (default_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi)
#     u_null = u_null[:, :7]
#     u_null = mm @ u_null
#     u += (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
#     return u.squeeze(-1)