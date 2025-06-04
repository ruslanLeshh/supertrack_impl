
import torch
def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.
    """
    ax, ay, az, aw = torch.unbind(a, -1)
    bx, by, bz, bw = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ox, oy, oz, ow), -1)

def quaternion_inverse(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of a batch of quaternions in [x, y, z, w] format.
    Supports both single quaternions [4] and batches [B, 4].
    """
    q = quaternion.clone()
    q[..., :3] = -q[..., :3]  # Negate x, y, z
    norm_sq = torch.sum(quaternion * quaternion, dim=-1, keepdim=True)  # [B, 1] or [1]
    return q / norm_sq  # Broadcast-safe


def quaternion_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Apply the rotation given by a quaternion to a 3D point. same as p.rotateVector(q,v)
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((point, real_parts), -1)  
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_inverse(quaternion),
    )
    return out[..., :3]

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4) in [x, y, z, w] format.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    # Unpack components assuming [x, y, z, w] format
    i, j, k, r = torch.unbind(quaternions, -1)
    
    # Scale factor
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    # Compute rotation matrix components
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))













def to_local_t(state): #
    """
    all_pos  
    all_vel
    all_rot
    all_rot_vel
    """
    local_pos = []  
    local_vel = []
    local_rot = []
    local_rot_vel = []
    local_height = []
    inv_root_wr = quaternion_inverse(state['rot'][0])
    for i in range(len(state['pos'])):
        transformed_pos = quaternion_apply(inv_root_wr, (state['pos'][i] - state['pos'][0]))
        local_pos.append(transformed_pos)

        transformed_vel = quaternion_apply(inv_root_wr, state['vel'][i]) # grad
        local_vel.append(transformed_vel)

        transformed_rot = quaternion_raw_multiply(inv_root_wr, state['rot'][i]) # quaternion mul
        transformed_rot = quaternion_to_matrix(transformed_rot)
        transformed_rot = transformed_rot.reshape(3, 3)
        transformed_rot = transformed_rot[:, [0, 2]].flatten() # two-axis rotation matrix as 6x vector
        local_rot.append(transformed_rot)
        # print('\nROT_ROT_ROT',transformed_rot)
        transformed_rot_vel = quaternion_apply(inv_root_wr, state['rot_vel'][i]) # grad
        local_rot_vel.append(transformed_rot_vel)

        link_height = state['pos'][i][1]
        local_height.append(link_height)

        up_vec = quaternion_apply(inv_root_wr, torch.tensor([0, 1, 0])) 
    return [torch.stack(local_pos), torch.stack(local_vel), torch.stack(local_rot), torch.stack(local_rot_vel), torch.stack(local_height), up_vec]


def quat_exp_t(v: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    Compute exponential map from vector to quaternion in [x, y, z, w] format.
    v = θ * a (angle * axis)
    expects halfangle (v*0.5), so that quaternion represents original angle
    """
    # Extract angle (magnitude of the vector)
    halfangle = torch.norm(v, dim=-1, keepdim=True)  # (..., 1)
    # print('halfangle    ',halfangle)
    # Handle small angles safely
    small_angle = halfangle < eps  # (..., 1)
    norm_v = torch.where(small_angle, torch.ones_like(halfangle), halfangle)

    # Scaled sine term (sin(halfangle) / halfangle)
    s = torch.sin(halfangle) / norm_v  # (..., 1)
    c = torch.cos(halfangle)  # (..., 1)

    # Ensure `s` has the same shape as `v` for broadcasting
    vec_part = s * v  # (..., 3)

    # Concatenate vector part and scalar part
    quat = torch.cat([vec_part, c], dim=-1)  # (..., 4)

    return quat



def quat_exp(v, eps=1e-8):
    """
    Exponential map from axis-angle to quaternion.
    """
    norm = np.linalg.norm(v, axis=-1, keepdims=True)  # shape (..., 1)
    
    # Avoid division by zero
    small_angle = norm < eps
    norm_safe = np.where(small_angle, 1.0, norm)
    # print('halfangle    ',norm)
    s = np.sin(norm) / norm_safe
    c = np.cos(norm)

    quat_xyz = s * v
    quat_w = c

    # If small angle, use identity quaternion
    quat_xyz = np.where(small_angle, v, quat_xyz)
    quat_w = np.where(small_angle.squeeze(-1), 1.0, quat_w.squeeze(-1))

    return np.concatenate([quat_xyz, quat_w[..., np.newaxis]], axis=-1)


def quat_log_t(q: torch.Tensor, eps: float = 1e-8):
    vec = q[..., :3]              # Vector part (x, y, z)
    scalar = q[..., 3]            # Scalar part (w)
    
    length = torch.norm(vec, dim=-1, keepdim=True)  # Shape: [..., 1]

    # Avoid divide-by-zero: use torch.where
    safe_vec = vec / (length + eps)                 # Normalized vector part
    halfangle = torch.atan2(length.squeeze(-1), scalar)  # Shape: [...]

    log = halfangle.unsqueeze(-1) * safe_vec        # Shape: [..., 3]
    return torch.where(length > eps, log, vec)      # If length < eps, just return vec


# APPROXIMATIONS

import torch

def quat_exp_approx_t(v: torch.Tensor) -> torch.Tensor:
    """
    Approximate quaternion exponential map.
    Returns a normalized quaternion [x, y, z, w].
    """
    q = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)  # [x, y, z, w]
    return torch.nn.functional.normalize(q, dim=-1)  # Normalize the quaternion

def quat_log_approx_t(q: torch.Tensor) -> torch.Tensor:
    """
    Approximate the logarithm of a quaternion.
    Assumes small rotations, so it ignores normalization and simplifies to (x, y, z).
    """
    return q[..., :3]  # Extracts (x, y, z), ignoring w

import numpy as np
def quat_exp_approx(v: np.ndarray) -> np.ndarray:
    """
    Approximate quaternion exponential map using NumPy.
    Returns a normalized quaternion [x, y, z, w].
    """   
    ones = np.ones((v.shape[0], 1))
    q = np.concatenate([v, ones], axis=1) # [x, y, z, w]
    return q / np.linalg.norm(q, axis=1, keepdims=True)  # Normalize the quaternion
















def to_local_git(pos,vel,rot,ang,batch_size, device): #
    """
    pos  
    vel
    rot
    ang
    """


    inv_root_wr = quaternion_inverse(rot[:, 0:1, :]) #([frame, joints, quaternion])
    # print(inv_root_wr)
    l_pos = quaternion_apply(inv_root_wr, (pos - pos[:, 0:1, :])) 

    l_vel = quaternion_apply(inv_root_wr, vel)

    matrix = quaternion_to_matrix( quaternion_raw_multiply(inv_root_wr, rot)) # #[:, :, 4] ->[:, :, 3, 3]
    batch_dim = matrix.size()[:-2] 
    l_rot = matrix[..., :, [0, 2]].clone().reshape(batch_dim + (6,)) # [:, :, 6] take x and y and flat them as 6x vector
    
    l_ang = quaternion_apply(inv_root_wr, ang)

    l_height = pos[:, :, 1]

    up_vec = torch.tensor([0, 1, 0], dtype=torch.float32, device=device).expand(batch_size, 1, 3)  # Shape: (BS, 1, 3)

    l_up = quaternion_apply(inv_root_wr, up_vec).squeeze(1)
    # print(l_rot.shape)
    
    return l_pos, l_vel, l_rot, l_ang, l_height, l_up
