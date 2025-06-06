#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2
import copy

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    # ------------------------------------------------------------------
    # Convenience: update intrinsic world‑view transforms after changing
    # extrinsics (R / T) or global scene transform (trans / scale).
    # ------------------------------------------------------------------
    def set_pose(self, R: torch.Tensor, T: torch.Tensor,
                 trans=None, scale=None):
        """
        Update camera extrinsics and recompute all dependent matrices.

        Parameters
        ----------
        R : (3,3) torch tensor             – new rotation matrix
        T : (3,)  torch tensor             – new translation vector
        trans : (3,) np.ndarray OR torch   – global translation  (optional)
        scale : float                      – global scale        (optional)
        """
        self.R = R
        self.T = T
        if trans is not None:
            self.trans = np.array(trans) if isinstance(trans, (list, tuple)) else trans
        if scale is not None:
            self.scale = scale

        # Recompute transforms
        self.world_view_transform = torch.tensor(
            getWorld2View2(self.R.cpu().numpy(),
                           self.T.cpu().numpy(),
                           self.trans,
                           self.scale),
            dtype=torch.float32,
            device=self.data_device).T  # .transpose(0,1)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]



# --- Quaternion helpers & camera‑pose interpolation --------------------
import torch

def _rot_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    3×3 rotation matrix -> (w,x,y,z) unit quaternion
    """
    q = torch.empty(4, device=R.device, dtype=R.dtype)
    trace = R.trace()
    if trace > 0.0:
        s = torch.sqrt(trace + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    else:
        i = torch.argmax(torch.diag(R))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = torch.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2.0
        q[i + 1] = 0.25 * s
        q[0] = (R[k, j] - R[j, k]) / s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    return q / q.norm()

def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    """
    Spherical‑linear interpolation between two unit quaternions.
    """
    dot = torch.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    DOT_THR = 0.9995
    if dot > DOT_THR:
        return (1.0 - t) * q0 + t * q1
    theta_0 = torch.acos(dot)
    theta = theta_0 * t
    sin_t0 = torch.sin(theta_0)
    return (torch.sin(theta_0 - theta) * q0 + torch.sin(theta) * q1) / sin_t0

def interpolate_camera(ref_cam, tgt_cam, alpha: float = 0.5):
    """
    Returns a *clone* of ref_cam whose pose is interpolated toward tgt_cam
    by factor `alpha` (0 => ref, 1 => tgt).  Works even if .R/.T are NumPy.
    """
    device = "cuda"

    def as_tensor(x):
        return torch.as_tensor(x, device=device, dtype=torch.float32)

    R_ref = as_tensor(ref_cam.R)
    R_tgt = as_tensor(tgt_cam.R)
    t_ref = as_tensor(ref_cam.T)
    t_tgt = as_tensor(tgt_cam.T)

    q_ref = _rot_to_quat(R_ref)
    q_tgt = _rot_to_quat(R_tgt)
    q_mid = _quat_slerp(q_ref, q_tgt, alpha)

    w, x, y, z = q_mid
    R_mid = torch.tensor([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                          [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                          [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]],
                         device=device, dtype=torch.float32)

    cam = copy.deepcopy(ref_cam)
    new_T = (1.0 - alpha) * t_ref + alpha * t_tgt
    cam.set_pose(R_mid, new_T)
    return cam