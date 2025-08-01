import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from concurrent.futures.process import ProcessPoolExecutor

import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pprint import pprint

import cv2
import matplotlib.pyplot as plt
import pandas as pd

#import preset
# from utility.plot_tools import plt_imgshow, plt_show, plt_multi_imgshow
# from utility.torch_tools import str_tensor_shape
# from utility.xprint import pbox

from torch.nn import functional as F
import zipfile

import requests
from tqdm import tqdm
import torch

#from e_preprocess_scripts.aria_adt.aria_const import fpath_index_json, dpath_cache_aria_adt
#from utility.fctn import read_json, save_tensor, save_image, save_jsonl, read_jsonl, load_tensor, save_pickle, read_pickle
import numpy as np
import os
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataProvider,
    AriaDigitalTwinSkeletonProvider,
    AriaDigitalTwinDataPathsProvider,
    bbox3d_to_line_coordinates,
    bbox2d_to_image_coordinates,
)
from projectaria_tools.core import calibration

from scipy.spatial.transform import Rotation as R

# from e_preprocess_scripts.b7_preprocess_aria_adt_imu import DatasetADT

from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import numpy as np
from PIL import Image

import torchvision.transforms as T

depth_cnn_path = '/home/xth/vio_seg/yfz/depthCNN'
depth_cnn_path = '/home/external/ORB_SLAM3_VIO/depthCNN'

# 为了确保路径的正确性，最好再检查一下
if not os.path.exists(depth_cnn_path):
    raise FileNotFoundError(f"指定的路径不存在: {depth_cnn_path}")

# 将这个路径临时添加到Python解释器的模块搜索路径列表中
sys.path.append(depth_cnn_path)
# --- 步骤 2: 导入函数 ---
# 现在因为路径已经设置好了，你可以直接像导入本地模块一样导入函数
try:
    from evaluate_demo import use_depth_predictor
    print("成功导入 use_depth_predictor 函数。")
except ImportError as e:
    print(f"导入失败，请检查'evaluate_demo.py'文件是否存在于'{depth_cnn_path}'中，并且函数名'use_depth_predictor'是否正确。")
    # 如果导入失败，最好退出程序
    sys.exit(1)

skey='Apartment_release_clean_seq131_M1292'

indexinfo = read_json(fpath_index_json)

sequences = indexinfo['sequences']
stream_id_str = "214-1"
stream_id = StreamId(stream_id_str)

scratch_path = './scratch'

def get_timestamp_ns(skey, filename='2d_bounding_box.csv', colkey='timestamp[ns]'):
    dpath_seq = os.path.join(dpath_cache_aria_adt, skey)
    os.makedirs(dpath_seq, exist_ok=True)
    fpath_csv = os.path.join(dpath_seq, filename)

    df = pd.read_csv(fpath_csv)

    df = df[df['stream_id'] == stream_id_str]

    ts_ns = df[colkey]

    res = list(sorted(set(ts_ns.tolist())))
    return res

def get_gt_provider(skey):
    dpath_seq = os.path.join(dpath_cache_aria_adt, skey)
    os.makedirs(dpath_seq, exist_ok=True)
    paths_provider = AriaDigitalTwinDataPathsProvider(dpath_seq)
    data_paths = paths_provider.get_datapaths()
    print("loading ground truth data...")
    gt_provider = AriaDigitalTwinDataProvider(data_paths)
    return gt_provider



def load_img_depth(gt_provider, select_timestamps_ns, src_calib, dst_calib):
    #===========image===========#
    # fetch the raw data 
    print("loading image with timestamp: ", select_timestamps_ns, " ns")
    image_with_dt = gt_provider.get_aria_image_by_timestamp_ns(select_timestamps_ns, stream_id)

    # check image is valid. It's always possible that the data retrieval fails, therefore all 
    # returned data not only contains dt, but also contains an is_valid() function, or returns 
    # an optional variable. 
    assert image_with_dt.is_valid(), "Image not valid!"    

    # since all image data and some GT data are discrete, each time we query discrete data 
    # from the data providers we return an object with dt. Dt is the difference between 
    # returned data time and the query time. For this case, since we are querying for images
    # using the real image times we already fetched, dt should be zero. Note that if we iterated
    # through GT timestamps, they may not correspond to the exact same time points as the camera 
    # data. We align GT data to SLAM camera timestamps, so if querying RGB images using GT 
    # timestamps, dt will not be zero
    print("image_time - query_time: ", image_with_dt.dt_ns(), " ns")

    # convert image to numpy array
    image = image_with_dt.data().to_numpy_array()
    
    print('image shape ', image.shape)
    print('image type ', image.dtype)
    # rectify image
    image = calibration.distort_by_calibration(image, dst_calib, src_calib)

    image_np = image
    # pad SLAM camera gray-scale image to 3 channel for color visualization
    image = np.repeat(image[..., np.newaxis], 3, axis=2) if len(image.shape) < 3 else image
    #plt.imshow(image); plt.xticks([]); plt.yticks([]); 
    # OpenCV 需要 BGR 格式，因此需要从 RGB 转换
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{scratch_path}/frame_{select_timestamps_ns//1e7}.png", image_bgr)

    #===========seg===========#
    segmentation_with_dt = gt_provider.get_segmentation_image_by_timestamp_ns(select_timestamps_ns, stream_id)
    # check if the result is valid
    assert segmentation_with_dt.is_valid(), "segmentation not valid for input timestamp!"
    print("groundtruth_time - query_time = ", segmentation_with_dt.dt_ns(), "ns")
    segmentation_for_viz = segmentation_with_dt.data().get_visualizable().to_numpy_array()

    print('segmentation_for_viz shape ', segmentation_for_viz.shape)
    print('segmentation_for_viz type ', segmentation_for_viz.dtype)
    print('segmentation_for_viz max ', segmentation_for_viz.max())

    segmentation_for_viz = calibration.distort_by_calibration(segmentation_for_viz, dst_calib, src_calib)

    segmentation_for_viz_bgr = cv2.cvtColor(segmentation_for_viz, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{scratch_path}/seg_{select_timestamps_ns//1e7}.png", segmentation_for_viz_bgr)
    segmentation_data = segmentation_with_dt.data().to_numpy_array()
    # print('segmentation_data ', segmentation_data.shape)
    segmentation_data = segmentation_for_viz #np.repeat(segmentation_for_viz[..., np.newaxis], 3, axis=2)



    #===========depth===========#
    depth_with_dt = gt_provider.get_depth_image_by_timestamp_ns(select_timestamps_ns, stream_id)

    # check if the result is valid
    if not depth_with_dt.is_valid():
        print("depth map not valid for input timestamp!")
    print("groundtruth_time - query_time = ", depth_with_dt.dt_ns(), "ns")

    # draw image
    depth_image = depth_with_dt.data()
    depth_for_vis = depth_with_dt.data().get_visualizable().to_numpy_array()
    depth_for_vis = np.repeat(depth_for_vis[..., np.newaxis], 3, axis=2)
    depth_for_vis_bgr = cv2.cvtColor(depth_for_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{scratch_path}/depth_{select_timestamps_ns//1e7}.png", depth_for_vis_bgr)

    row = int(depth_image.get_height() / 3) 
    col = int(depth_image.get_width() / 2)
    depth_mm = depth_image.at(col, row)
    print(f'depth_mm in row {row} col {col}: {depth_mm}')

    # convert depth to numpy array
    depth_image_np = depth_with_dt.data().to_numpy_array()

    # rectify image
    depth_image_np = calibration.distort_by_calibration(depth_image_np, dst_calib, src_calib)

    return image_np, depth_image_np, segmentation_data

def load_translate_quert(target_ms, aria_trajectory_path):
    df = pd.read_csv(aria_trajectory_path)
    # 创建新的 column: tracking_timestamp_ms
    df["timestamp_ms"] = (df["tracking_timestamp_us"] // 10000).astype(int)
    row = df[df["timestamp_ms"] == target_ms]
    if row.empty:
        print("没有找到匹配的行")
    else:
        # 提取目标列
        result = row[[
            "tx_world_device", "ty_world_device", "tz_world_device", "qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"
        ]].values[0]  # 取第一行的值作为 numpy 数组

        print("对应位姿：", result)
        return result

def build_pose_matrix(pos):
    angle_deg=0.7
    axis='y'
    q = pos[3:]  # [qx, qy, qz, qw]
    t = pos[:3]  # [tx, ty, tz]
    #rot = R.from_quat(q).as_matrix()
    # 2. 从原始四元数创建初始旋转
    rot_initial = R.from_quat(q)

    # 用于返回的轴和角
    random_axis = None
    used_angle_deg = 0

    # 3. 如果指定了旋转角度，则创建并复合一个随机旋转
    if angle_deg != 0.0:
        # ---- 这是核心部分 ----
        # 3a. 生成一个随机的三维向量并归一化，作为旋转轴
        random_axis = np.random.randn(3) # 从标准正态分布中采样
        random_axis /= np.linalg.norm(random_axis) # 归一化为单位向量

        # 3b. 将角度从度转换为弧度
        angle_rad = np.deg2rad(angle_deg)
        used_angle_deg = angle_deg

        # 3c. 创建旋转向量 (axis * angle)
        rot_vec = random_axis * angle_rad

        # 3d. 从旋转向量创建额外的旋转对象
        rot_additional = R.from_rotvec(rot_vec)

        # 3e. 复合旋转
        rot_final = rot_additional * rot_initial
        # ---------------------
    else:
        # 如果没有额外旋转，则最终旋转就是初始旋转
        rot_final = rot_initial
    # 4. 将最终的旋转对象转换为旋转矩阵
    rot = rot_final.as_matrix()


    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = t

    # T1 = np.eye(4)
    # T1[:3, :3] = rot.T
    # T2 = np.eye(4)
    # T2[:3, 3] = -t
    # T=T1@T2
    # T=np.linalg.inv(T)
    return T

def warp_rgb_to_new_view(rgb, depth_mm, pos, next_pos, K):
    h, w, c = rgb.shape

    # Convert depth to meters
    if depth_mm is None:
        depth = np.ones_like(rgb)[:,:,0]
    else:
        depth = depth_mm.astype(np.float32) / 1000.0

    # Build camera intrinsics
    K = np.array(K)
    K_inv = np.linalg.inv(K)

    cam_device_pos = [-0.00428072, -0.0118417, -0.00511398, 0.32414, 0.0402123, 0.0410768, 0.944261]
    T_device_cam = build_pose_matrix(cam_device_pos)
    # Compute poses
    T_curr = build_pose_matrix(pos)
    T_next = build_pose_matrix(next_pos)
    T_rel = np.linalg.inv(T_device_cam) @ np.linalg.inv(T_next) @ T_curr @ T_device_cam
    #T_rel = T_next @ np.linalg.inv(T_curr)

    # Generate pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)  # (N, 3)
    depth_flat = depth.reshape(-1)
    # print('max depth ', max(depth_flat))
    # print('depth ', depth_flat.shape)
    #depth_flat = 1

    # Unproject to 3D
    pts_cam = (K_inv @ uv1.T * depth_flat).T  # (N, 3)
    pts_cam_homo = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)

    # Transform to next camera frame
    pts_next = (T_rel @ pts_cam_homo.T).T[:, :3]

    # Project back to image
    proj = (K @ pts_next.T).T
    proj /= proj[:, 2:3]
    #print('proj shape ', proj.shape)
    u_proj = np.round(proj[:, 0]).astype(int)
    v_proj = np.round(proj[:, 1]).astype(int)
    #print('u_proj shape ', u_proj.shape)
    # Prepare output image
    warped = np.zeros_like(rgb)
    valid = (u_proj >= 0) & (u_proj < w) & (v_proj >= 0) & (v_proj < h)
    #print('valid shape ', valid.shape)
    warped[v_proj[valid], u_proj[valid]] = rgb.reshape(-1, 3)[valid]

    # # Step 5: remap
    # u_next = proj[:, 0].reshape(h, w).astype(np.float32)
    # v_next = proj[:, 1].reshape(h, w).astype(np.float32)
    # warped = cv2.remap(img1, u_next, v_next, interpolation=cv2.INTER_LINEAR,
    #                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return warped

def warp_rgb_to_new_view_old(rgb, depth_mm, pos, next_pos, K):
    h, w, c = rgb.shape

    # --- 1. Setup: Convert depth and get matrices ---
    if depth_mm is None:
        # If no depth, assume a constant depth of 1 meter for testing
        depth = np.ones((h, w), dtype=np.float32)
    else:
        depth = depth_mm.astype(np.float32) / 1000.0  # Convert depth to meters

    K = np.array(K)
    K_inv = np.linalg.inv(K)

    # --- 2. Compute Relative Pose (with the likely FIX) ---
    T_curr = build_pose_matrix(pos)
    T_next = build_pose_matrix(next_pos)

    # This assumes T_curr and T_next are "view matrices" (world-to-camera)
    # which is the most common source of this kind of error.
    T_rel = np.linalg.inv(T_next) @ T_curr
    #T_rel = T_next @ np.linalg.inv(T_curr)

    # --- 3. Unproject source pixels to 3D points in its own camera frame ---
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1) # Shape: (h, w, 3)

    # Unproject: P_cam = depth * K_inv @ uv
    # Reshape for matrix multiplication
    uv1_flat = uv1.reshape(-1, 3).T  # Shape: (3, h*w)
    depth_flat = depth.reshape(-1)  # Shape: (h*w,)
    pts_cam = (K_inv @ uv1_flat) * depth_flat  # Shape: (3, h*w)

    # --- 4. Transform points to the next camera's frame ---
    pts_cam_homo = np.vstack([pts_cam, np.ones_like(depth_flat)]) # Shape: (4, h*w)
    pts_next = (T_rel @ pts_cam_homo)[:3, :] # Shape: (3, h*w)

    # --- 5. Project 3D points onto the next camera's image plane ---
    proj_homo = K @ pts_next # Shape: (3, h*w)

    # Perspective divide (handle depths close to zero to avoid errors)
    z = proj_homo[2, :]
    z[z == 0] = 1e-6 # Avoid division by zero
    u_proj = proj_homo[0, :] / z
    v_proj = proj_homo[1, :] / z

    # Reshape projected coordinates into a map for cv2.remap
    map_x = u_proj.reshape(h, w).astype(np.float32)
    map_y = v_proj.reshape(h, w).astype(np.float32)

    # --- 6. Use cv2.remap for high-quality inverse warping ---
    # This samples from the source 'rgb' image using the calculated map
    warped_image = cv2.remap(
        src=rgb,
        map1=map_x,
        map2=map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return warped_image

def build_pose_matrix_torch(pos):
    """
    构建齐次变换矩阵，支持多批次处理。
    假设输入 pos 的形状为 (B, 7)，其中 B 是批次大小，
    每一行的前 3 个元素是平移 (tx, ty, tz)，后 4 个元素是四元数 (qx, qy, qz, qw)。

    注意：PyTorch 本身没有直接将四元数转换为旋转矩阵的内置函数，
    因此这里仍然使用 scipy.spatial.transform.Rotation 进行转换，
    但会在批次维度上进行处理。

    Args:
        pos (torch.Tensor): 相机位姿，形状为 (B, 7)。

    Returns:
        torch.Tensor: 齐次变换矩阵，形状为 (B, 4, 4)。
    """
    B = pos.shape[0]
    q_numpy = pos[:, 3:].cpu().numpy()  # (B, 4)
    t_numpy = pos[:, :3].cpu().numpy()  # (B, 3)

    rot_matrices_numpy = np.array([R.from_quat(quat).as_matrix() for quat in q_numpy])  # (B, 3, 3)
    rot_matrices_torch = torch.from_numpy(rot_matrices_numpy).float().to(pos.device)
    t_torch = pos[:, :3].unsqueeze(-1)  # (B, 3, 1)

    T = torch.eye(4, dtype=torch.float32, device=pos.device).unsqueeze(0).repeat(B, 1, 1)  # (B, 4, 4)
    T[:, :3, :3] = rot_matrices_torch
    T[:, :3, 3] = t_torch.squeeze(-1)

    return T

def warp_rgb_to_new_view_torch(rgb, depth_m, pos, next_pos, K):
    """
    将 RGB 图像从当前视角扭曲到下一个视角，支持多批次处理。

    Args:
        rgb (torch.Tensor): 输入 RGB 图像，形状为 (B, H, W, C) 或 (B, C, H, W)。
                             推荐使用 (B, C, H, W) 以符合 PyTorch 惯例。
        depth_mm (torch.Tensor): 深度图（毫米），形状为 (B, H, W)。
                                  如果为 None，则假设所有深度为 1 米。
        pos (torch.Tensor): 当前相机的齐次变换矩阵，形状为 (B, 4, 4)。
        next_pos (torch.Tensor): 下一个相机的齐次变换矩阵，形状为 (B, 4, 4)。
        K (torch.Tensor): 相机内参矩阵，形状为 (B, 3, 3)。

    Returns:
        torch.Tensor: 扭曲后的 RGB 图像，形状与输入 rgb 相同。
    """
    # 确保 rgb 是 (B, C, H, W) 格式，如果不是则进行转换
    if rgb.shape[1] == 3 or rgb.shape[1] == 1: # Assuming C is 3 or 1
        # It's already (B, C, H, W)
        pass
    elif rgb.shape[-1] == 3 or rgb.shape[-1] == 1: # Assuming C is last dim
        # Convert (B, H, W, C) to (B, C, H, W)
        rgb = rgb.permute(0, 3, 1, 2)
    else:
        raise ValueError("Unsupported RGB image format. Expected (B, H, W, C) or (B, C, H, W).")

    B, C, H, W = rgb.shape
    device = rgb.device
    dtype = rgb.dtype
    print('dtype ', dtype)

    # 1. 转换深度到米
    if depth_m is None:
        # 如果没有深度图，假设所有深度为 1 米
        depth = torch.ones(B, H, W, dtype=dtype, device=device)
    else:
        depth = depth_m.to(dtype) # 单位是 米

    # 2. 构建相机内参逆矩阵
    K_inv = torch.linalg.inv(K) # (B, 3, 3)
    print('k inv dtype ', {K_inv.dtype})

    # 3. 计算相对姿态变换矩阵
    # T_rel = T_next_inv @ T_curr
    # 这里的 @ 是 PyTorch 的矩阵乘法运算符
    T_curr = build_pose_matrix_torch(pos)
    T_next = build_pose_matrix_torch(next_pos)
    T_rel = torch.linalg.inv(T_next) @ T_curr # (B, 4, 4)
    print('T_rel dtype ', {T_rel.dtype})
    # 4. 生成像素网格 (当前图像坐标)
    # 创建 (H, W) 的像素坐标网格
    # v 对应行 (height), u 对应列 (width)
    v_coords, u_coords = torch.meshgrid(torch.arange(H, device=device),
                                        torch.arange(W, device=device),
                                        indexing='ij') # indexing='ij' for (row, col)
    
    # 将 u, v 转换为齐次坐标 (u, v, 1)
    # 形状从 (H, W) 变为 (H*W, 3)
    uv1 = torch.stack([u_coords, v_coords, torch.ones_like(u_coords)], dim=-1) # (H, W, 3)
    uv1_flat = uv1.reshape(H * W, 3).to(dtype) # (H*W, 3)

    # 将深度图展平
    depth_flat = depth.reshape(B, H * W) # (B, H*W)

    # 5. 反投影到 3D 空间 (当前相机坐标系)
    # K_inv @ uv1.T 是 (B, 3, 3) @ (3, H*W) -> (B, 3, H*W)
    # depth_flat 是 (B, H*W)
    # 需要将 depth_flat 扩展一个维度进行广播乘法
    print('uv1_flat dtype ', uv1_flat.dtype)
    pts_cam = (K_inv @ uv1_flat.T.unsqueeze(0)) * depth_flat.unsqueeze(1) # (B, 3, H*W)
    
    # 转换为齐次坐标 (X, Y, Z, 1)
    ones_batch = torch.ones(B, 1, H * W, dtype=dtype, device=device) # (B, 1, H*W)
    pts_cam_homo = torch.cat([pts_cam, ones_batch], dim=1) # (B, 4, H*W)

    # 6. 变换到下一个相机坐标系
    # T_rel 是 (B, 4, 4)， pts_cam_homo 是 (B, 4, H*W)
    pts_next_homo = T_rel @ pts_cam_homo # (B, 4, H*W)
    pts_next = pts_next_homo[:, :3, :] # (B, 3, H*W)

    # 7. 投影回图像平面 (下一个相机坐标系)
    # K 是 (B, 3, 3)， pts_next 是 (B, 3, H*W)
    proj = K @ pts_next # (B, 3, H*W)
    
    # 归一化齐次坐标 (除以 Z)
    # 避免除以零，添加一个小 epsilon
    proj_z = proj[:, 2:3, :] # (B, 1, H*W)
    # 确保 Z 不为零，避免 NaN
    proj_z = torch.where(proj_z == 0, torch.full_like(proj_z, 1e-6), proj_z) 
    
    proj_normalized = proj / proj_z # (B, 3, H*W)

    # 提取投影后的 u, v 坐标
    u_proj = proj_normalized[:, 0, :].reshape(B, H, W) # (B, H, W)
    v_proj = proj_normalized[:, 1, :].reshape(B, H, W) # (B, H, W)

    # 8. 准备用于 grid_sample 的采样网格
    # grid_sample 需要的坐标范围是 [-1, 1]，其中 (-1,-1) 是左上角，(1,1) 是右下角
    # u_proj, v_proj 是像素坐标 (0 到 W-1 或 H-1)
    
    # 将像素坐标归一化到 [-1, 1]
    # x 对应 u 坐标 (宽度), y 对应 v 坐标 (高度)
    grid_x = (u_proj / (W - 1)) * 2 - 1 # (B, H, W)
    grid_y = (v_proj / (H - 1)) * 2 - 1 # (B, H, W)

    # 堆叠 x 和 y 坐标，形成 (B, H, W, 2) 的网格
    grid = torch.stack([grid_x, grid_y], dim=-1) # (B, H, W, 2)

    # 9. 使用 grid_sample 进行图像扭曲
    # rgb 已经是 (B, C, H, W) 格式
    # mode='bilinear' 是双线性插值，通常效果更好
    # padding_mode='zeros' 表示超出边界的像素用 0 填充
    # align_corners=True 通常用于图像处理任务，与 OpenCV remap 行为更接近
    warped_rgb = F.grid_sample(rgb, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    # 如果原始 rgb 是 (B, H, W, C) 格式，并且希望输出也是该格式，则进行转换
    # if rgb.shape[1] == C_original: # Check if original C was last dim
    #     warped_rgb = warped_rgb.permute(0, 2, 3, 1)

    return warped_rgb

def try_transform():
    #sequence_path = os.path.join(dpath_cache_aria_adt, skey) #'/mnt/extdisk/share_data/DriverD/b_data_train/data_b_cache/aria_adt/Apartment_release_clean_seq131_M1292'
    sequence_path = '/mnt/ssd_ext/share_data/DriverD/b_data_train/data_b_cache/aria_adt/Apartment_release_clean_seq131_M1292'
    paths_provider = AriaDigitalTwinDataPathsProvider(sequence_path)
    all_device_serials = paths_provider.get_device_serial_numbers()
    sequence_name = os.path.basename(sequence_path)
    print("all devices for sequence ", sequence_name, ":")
    for idx, device_serial in enumerate(all_device_serials):
        print("device number - ", idx, ": ", device_serial)

    selected_device_number = 0
    data_paths = paths_provider.get_datapaths_by_device_num(selected_device_number)
    print(data_paths)

    print("loading ground truth data...")
    gt_provider = AriaDigitalTwinDataProvider(data_paths)
    print("done loading ground truth data")

    stream_id = StreamId("214-1")
    img_timestamps_ns = gt_provider.get_aria_device_capture_timestamps_ns(stream_id)
    print("There are {} frames".format(len(img_timestamps_ns)))
    print()

    ### Rectify the RGB image
    # get source calibration - Aria original camera model
    sensor_name = gt_provider.raw_data_provider_ptr().get_label_from_stream_id(stream_id)
    print('sensor_name ', sensor_name)
    device_calib = gt_provider.raw_data_provider_ptr().get_device_calibration()
    src_calib = device_calib.get_camera_calib(sensor_name)

    # create output calibration: a pinhole rectified image size 512x512 and focal length 280
    dst_calib = calibration.get_linear_camera_calibration(1408, 1408, 1408/352*152, sensor_name)
    print('src_calib ', src_calib)
    print('dst_calib ', dst_calib)

    # # rectify image
    # rectified_image = calibration.distort_by_calibration(image, dst_calib, src_calib)

    # choose the frame in the middle of the sequence
    select_timestamps_ns = img_timestamps_ns[int(len(img_timestamps_ns)/2)]
    print('select_timestamps_ns ', select_timestamps_ns)
    next_select_timestamps_ns = img_timestamps_ns[int(len(img_timestamps_ns)/2)+1]
    print('next_select_timestamps_ns ', next_select_timestamps_ns)

    img1, depth1, seg1 = load_img_depth(gt_provider, select_timestamps_ns, src_calib, dst_calib)
    img2, depth2, seg2 = load_img_depth(gt_provider, next_select_timestamps_ns, src_calib, dst_calib)
    print('seg shape ', seg1.shape)
    print('img1 shape ', img1.shape)
    aria_trajectory_path = os.path.join(sequence_path, 'aria_trajectory.csv')

    pos = load_translate_quert(int(select_timestamps_ns // 1e7), aria_trajectory_path)
    next_pos = load_translate_quert(int(next_select_timestamps_ns // 1e7), aria_trajectory_path)

    # === Step 1. 相机内参 K ===
    fx = fy = 152.73524
    cx, cy = 176, 176
    # 当前图像大小
    h_new, w_new = 1408, 1408

    # 原始分辨率
    h_ref, w_ref = 352, 352

    scale_x = w_new / w_ref
    scale_y = h_new / h_ref

    # 缩放相机内参
    fx = fx * scale_x
    fy = fy * scale_y
    cx = cx * scale_x
    cy = cy * scale_y

    K = [[fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]]

    warped_img = warp_rgb_to_new_view(img1, depth1, pos, next_pos, K)
    #warped_img = warp_rgb_to_new_view_remap_new(img1, depth1, pos, next_pos, K)
    #warped_img = warp_rgb_to_new_view(img1, depth1, pos, pos, K)
    warped_bgr = cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{scratch_path}/interp_img.png", warped_bgr)

    warped_seg = warp_rgb_to_new_view(seg1, depth1, pos, next_pos, K)
    #warped_img = warp_rgb_to_new_view_remap_new(img1, depth1, pos, next_pos, K)
    #warped_img = warp_rgb_to_new_view(img1, depth1, pos, pos, K)
    warped_seg_bgr = cv2.cvtColor(warped_seg, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{scratch_path}/interp_seg.png", warped_seg_bgr)

def save_depth_img(depth_image_meters, path):
    min_val = np.min(depth_image_meters)
    max_val = np.max(depth_image_meters)

    if max_val - min_val > 0:
        # 使用公式: new = (old - min) * 255 / (max - min)
        depth_visual = 255 * (depth_image_meters - min_val) / (max_val - min_val)
    else:
        # 如果图像深度都一样，则全为0
        depth_visual = np.zeros(depth_image_meters.shape, dtype=np.uint8)

    # 3. 转换为8位无符号整数 (uint8)，这是标准图像格式的要求
    depth_visual_uint8 = depth_visual.astype(np.uint8)

    # 4. 保存为PNG或JPG文件
    cv2.imwrite(path, depth_visual_uint8)

def try_np_transform():
    sequence_path = '/mnt/ssd_ext/share_data/DriverD/b_data_train/data_b_cache/aria_adt/Apartment_release_clean_seq131_M1292' #os.path.join(dpath_cache_aria_adt, skey)
    paths_provider = AriaDigitalTwinDataPathsProvider(sequence_path)
    selected_device_number = 0
    data_paths = paths_provider.get_datapaths_by_device_num(selected_device_number)
    gt_provider = AriaDigitalTwinDataProvider(data_paths)
    stream_id = StreamId("214-1")
    sensor_name = gt_provider.raw_data_provider_ptr().get_label_from_stream_id(stream_id)
    print('sensor_name ', sensor_name)
    device_calib = gt_provider.raw_data_provider_ptr().get_device_calibration()
    src_calib = device_calib.get_camera_calib(sensor_name)
    dst_calib = calibration.get_linear_camera_calibration(1408, 1408, 1408/352*152, sensor_name)
    print('src_calib ', src_calib)
    print('dst_calib ', dst_calib)
    imu_calib = device_calib.get_imu_calib("imu-left")
    print('imu_calib ', imu_calib)

    dpath = "/home/xth/vio_seg/zhy/DynamicFocus/e_preprocess_scripts/aria_adt/scratch/dataset"
    depth1 = torch.load(os.path.join(dpath, "depth", "2_depth.pt")).numpy()*1000 #convert to mm
    depth1 = calibration.distort_by_calibration(depth1, dst_calib, src_calib)
    #depth1 = np.transpose(depth1, (1, 0))
    print('depth shape ', depth1.shape)
    depth_save_path = os.path.join(dpath, "depth", "2_depth.png")
    save_depth_img(np.transpose(depth1/1000, (0, 1)), depth_save_path)

    pos = torch.load(os.path.join(dpath, "imu", "2_imu.pt")).numpy()
    next_pos = torch.load(os.path.join(dpath, "imu", "15_imu.pt")).numpy()
    print(f'pos {pos}')
    print(f'next_pos {next_pos}')
    img1 = cv2.imread(os.path.join(dpath, "img", "2_img.png"))
    print('img1 max ', img1.max())
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    print('img 1 shape ', img1.shape)
    print(type(img1))
    #img1 = np.transpose(img1, (1, 0, 2))
    img1 = calibration.distort_by_calibration(img1, dst_calib, src_calib)#.astype(np.float32)
    img_bgr_for_cv2 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    output_filename_cv2 = os.path.join(dpath, "img", "2_img_np_calib.png")
    cv2.imwrite(output_filename_cv2, img_bgr_for_cv2)
    print('img 1 shape ', img1.shape)
    print(type(img1))
    
    #img1 = np.transpose(img1, (1, 0, 2))

    seg1 = cv2.imread(os.path.join(dpath, "seg", "2_seg.png"))
    seg1 = cv2.cvtColor(seg1, cv2.COLOR_BGR2RGB)
    seg1 = np.transpose(seg1, (1, 0, 2))#.astype(np.uint8)
    #seg1 = calibration.distort_by_calibration(seg1, dst_calib, src_calib).astype(np.float32)

    # === Step 1. 相机内参 K ===
    fx = fy = 152.73524
    cx, cy = 176, 176
    # 当前图像大小
    h_new, w_new = 1408, 1408

    # 原始分辨率
    h_ref, w_ref = 352, 352

    scale_x = w_new / w_ref
    scale_y = h_new / h_ref

    # 缩放相机内参
    fx = fx * scale_x
    fy = fy * scale_y
    cx = cx * scale_x
    cy = cy * scale_y

    K = [[fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]]

    warped_img = warp_rgb_to_new_view(img1, depth1, pos, next_pos, K)
    #warped_img = np.transpose(warped_img, (1, 0, 2))


    #warped_img = warp_rgb_to_new_view_remap_new(img1, depth1, pos, next_pos, K)
    #warped_img = warp_rgb_to_new_view(img1, depth1, pos, pos, K)
    warped_bgr = cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{scratch_path}/np_interp_img.png", warped_bgr)

    warped_seg = warp_rgb_to_new_view(seg1, depth1, pos, next_pos, K)
    warped_seg = np.transpose(warped_seg, (1, 0, 2))
    #warped_img = warp_rgb_to_new_view_remap_new(img1, depth1, pos, next_pos, K)
    #warped_img = warp_rgb_to_new_view(img1, depth1, pos, pos, K)
    warped_seg_bgr = cv2.cvtColor(warped_seg, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{scratch_path}/np_interp_seg.png", warped_seg_bgr)

def try_torch_transform():
    dpath = "/home/xth/vio_seg/zhy/DynamicFocus/e_preprocess_scripts/aria_adt/scratch/dataset"
    depth1 = torch.load(os.path.join(dpath, "depth", "1_depth.pt")) #convert to mm
    print('depth shape ', depth1.shape)
    depth_save_path = os.path.join(dpath, "depth", "1_depth.png")
    save_depth_img(np.transpose(depth1.numpy(), (1, 0)), depth_save_path)

    pos = torch.load(os.path.join(dpath, "imu", "1_imu.pt"))
    next_pos = torch.load(os.path.join(dpath, "imu", "12_imu.pt"))
    print(f'pos {pos}')
    print(f'next_pos {next_pos}')
    img1 = cv2.imread(os.path.join(dpath, "img", "1_img.png"))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = np.transpose(img1, (1, 0, 2))
    img1 = torch.tensor(img1, dtype=torch.float32)
    seg1 = cv2.imread(os.path.join(dpath, "seg", "1_seg.png"))
    seg1 = cv2.cvtColor(seg1, cv2.COLOR_BGR2RGB)
    seg1 = np.transpose(seg1, (1, 0, 2))
    seg1 = torch.tensor(seg1, dtype=torch.float32)
    # === Step 1. 相机内参 K ===
    fx = fy = 152.73524
    cx, cy = 176, 176
    # 当前图像大小
    h_new, w_new = 1408, 1408

    # 原始分辨率
    h_ref, w_ref = 352, 352

    scale_x = w_new / w_ref
    scale_y = h_new / h_ref

    # 缩放相机内参
    fx = fx * scale_x
    fy = fy * scale_y
    cx = cx * scale_x
    cy = cy * scale_y

    K = torch.tensor([[fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]])

    warped_img = warp_rgb_to_new_view_torch(img1.unsqueeze(0), depth1.unsqueeze(0), pos.unsqueeze(0), next_pos.unsqueeze(0), K.unsqueeze(0)).numpy()
    warped_img = np.transpose(warped_img[0], (2, 1, 0))
    #warped_img = warp_rgb_to_new_view_remap_new(img1, depth1, pos, next_pos, K)
    #warped_img = warp_rgb_to_new_view(img1, depth1, pos, pos, K)
    warped_bgr = cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{scratch_path}/interp_img.png", warped_bgr)

    warped_seg = warp_rgb_to_new_view_torch(seg1.unsqueeze(0), depth1.unsqueeze(0), pos.unsqueeze(0), next_pos.unsqueeze(0), K.unsqueeze(0)).numpy()
    print('warped_seg shape ', warped_seg.shape)
    warped_seg = np.transpose(warped_seg[0], (2, 1, 0))
    #warped_img = warp_rgb_to_new_view_remap_new(img1, depth1, pos, next_pos, K)
    #warped_img = warp_rgb_to_new_view(img1, depth1, pos, pos, K)
    warped_seg_bgr = cv2.cvtColor(warped_seg, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{scratch_path}/interp_seg.png", warped_seg_bgr)

def save_segmentation_as_rgb(cached_seg, output_path="segmentation.png", color_map=None):
    """
    将形状为 (B, 1, H, W) 的分割 mask 保存为 RGB 图像，不同的数字代表不同颜色。

    Args:
        cached_seg (torch.Tensor): 分割 mask 张量。
        output_path (str): 保存图像的路径。
        color_map (dict, optional): 数字到 RGB 颜色元组的映射。
                                     如果不提供，则使用默认颜色映射。
    """
    B, _, H, W = cached_seg.shape
    segmentation_map = cached_seg.squeeze(1).cpu().numpy()  # 移除通道维度并转为 NumPy (B, H, W)

    if color_map is None:
        # --- Start of Replacement Block ---
        
        # Programmatically generate a large, diverse color map
        color_map = {
            0: (0, 0, 0),          # Black (often for background)
            1: (255, 255, 255),    # White
            2: (255, 0, 0),        # Red
            3: (0, 255, 0),        # Green
            4: (0, 0, 255),        # Blue
        }
        
        # Start generating other colors from index 5
        color_index = 5
        
        # Define levels for R, G, B to create a grid of colors
        # We avoid the lowest values (too dark) and highest (already used)
        # to ensure more visually distinct colors.
        levels = [51, 102, 153, 204, 255]
        
        # Use a set to avoid adding duplicate colors
        generated_colors = set(color_map.values())

        for r in levels:
            for g in levels:
                for b in levels:
                    # Create a new color tuple
                    new_color = (r, g, b)
                    
                    # Add the color if it's not already in our map
                    if new_color not in generated_colors:
                        color_map[color_index] = new_color
                        generated_colors.add(new_color)
                        color_index += 1

    for b in range(1):
        segmentation_image_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        for label, color in color_map.items():
            mask = (segmentation_map[b] == label)
            #print('mask shape ', mask.shape)
            segmentation_image_rgb[mask] = color

        image = Image.fromarray(segmentation_image_rgb)
        batch_output_path = output_path.replace(".png", f"_batch_{b}.png") if B > 1 else output_path
        image.save(batch_output_path)
        print(f"Batch {b} segmentation mask saved to: {batch_output_path}")


def run_dataset():
    sequence_path = '/mnt/ssd_ext/share_data/DriverD/b_data_train/data_b_cache/aria_adt/Apartment_release_clean_seq131_M1292' #os.path.join(dpath_cache_aria_adt, skey)
    paths_provider = AriaDigitalTwinDataPathsProvider(sequence_path)
    selected_device_number = 0
    data_paths = paths_provider.get_datapaths_by_device_num(selected_device_number)
    gt_provider = AriaDigitalTwinDataProvider(data_paths)
    stream_id = StreamId("214-1")
    sensor_name = gt_provider.raw_data_provider_ptr().get_label_from_stream_id(stream_id)
    print('sensor_name ', sensor_name)
    device_calib = gt_provider.raw_data_provider_ptr().get_device_calibration()
    src_calib = device_calib.get_camera_calib(sensor_name)
    dst_calib = calibration.get_linear_camera_calibration(1408, 1408, 1408/352*152, sensor_name)
    print('src_calib ', src_calib)
    print('dst_calib ', dst_calib)
    imu_calib = device_calib.get_imu_calib("imu-left")
    print('imu_calib ', imu_calib)

    bs = 1
    fx = fy = 152.73524
    cx, cy = 176, 176
    # 当前图像大小
    h_new, w_new = 1408, 1408

    # 原始分辨率
    h_ref, w_ref = 352, 352

    scale_x = w_new / w_ref
    scale_y = h_new / h_ref

    # 缩放相机内参
    fx = fx * scale_x
    fy = fy * scale_y
    cx = cx * scale_x
    cy = cy * scale_y

    # K = torch.tensor([[fx, 0, cx],
    #     [0, fy, cy],
    #     [0,  0,  1]], dtype=torch.float32)
    # K_batched = K.unsqueeze(0).repeat(bs, 1, 1)

    K_batched = [[fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]]

    output_dir = "/home/xth/vio_seg/zhy/DynamicFocus/e_preprocess_scripts/aria_adt/scratch/dataset"
    os.makedirs(output_dir, exist_ok=True)

    img_dir = os.path.join(output_dir, f"img")
    os.makedirs(img_dir, exist_ok=True)

    img_prev_dir = os.path.join(output_dir, f"img_prev")
    os.makedirs(img_prev_dir, exist_ok=True)

    seg_dir = os.path.join(output_dir, f"seg")
    os.makedirs(seg_dir, exist_ok=True)

    seg_prev_dir = os.path.join(output_dir, f"seg_prev")
    os.makedirs(seg_prev_dir, exist_ok=True)

    warped_rgb_dir = os.path.join(output_dir, f"warped_rgb")
    os.makedirs(warped_rgb_dir, exist_ok=True)

    warped_seg_dir = os.path.join(output_dir, f"warped_seg")
    os.makedirs(warped_seg_dir, exist_ok=True)

    cached_seg_dir = os.path.join(output_dir, f"cached_seg")
    os.makedirs(cached_seg_dir, exist_ok=True)

    cached_rgb_dir = os.path.join(output_dir, f"cached_rgb")
    os.makedirs(cached_rgb_dir, exist_ok=True)

    depth_dir = os.path.join(output_dir, f"depth")
    os.makedirs(depth_dir, exist_ok=True)

    imu_dir = os.path.join(output_dir, f"imu")
    os.makedirs(imu_dir, exist_ok=True)

    dataset = DatasetADT()
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)
    i = 0
    obj_id = 1
    iou_sum = 0
    for batch in tqdm(dataloader):
        fid, img_RGB_Bx3xHxW, F_HW_Bx2, seg_A_Bx1xHxW, Y_cls_Bx1, imu_info_Bx6, iid, prev_fid, prev_img_RGB_Bx3xHxW, prev_F_HW_Bx2, prev_seg_A_Bx1xHxW, prev_Y_cls_Bx1, prev_imu_info_Bx6, prev_iid, prev_depth_tensor_BxHxW = batch

        coords = F_HW_Bx2.unsqueeze(1)  # 将形状从 (B, 2) 变为 (B, 1, 2)
        # 将归一化坐标转换回像素坐标
        x_pixel = (coords[:, :, 0] * (1408 - 1) + 0.5).long().clamp(0, 1408 - 1)
        y_pixel = (coords[:, :, 1] * (1408 - 1) + 0.5).long().clamp(0, 1408 - 1)
        single_depth = prev_depth_tensor_BxHxW[0, int(y_pixel.item()), int(x_pixel.item())]
        print('single_depth ', single_depth)
        print('y_pixel ', y_pixel.item())
        print('x_pixel ', x_pixel.item())
        
        # TODO:depth prediction
        tensor_to_convert = prev_img_RGB_Bx3xHxW.squeeze(0)
        to_pil = T.ToPILImage()
        image_tensor = to_pil(tensor_to_convert)
        depth_tensor = use_depth_predictor(image_tensor, int(x_pixel.item()), int(y_pixel.item()))
        print('depth_tensor ', depth_tensor)
        # end 
        #single_depth = depth_tensor
        prev_depth_tensor_BxHxW = torch.ones_like(prev_depth_tensor_BxHxW) * single_depth
        prev_depth_tensor_BxHxW_np = prev_depth_tensor_BxHxW[0].numpy()*1000
        prev_depth_tensor_BxHxW_np = prev_depth_tensor_BxHxW_np #calibration.distort_by_calibration(prev_depth_tensor_BxHxW_np, dst_calib, src_calib)

        prev_imu_info_Bx6_np = prev_imu_info_Bx6[0].numpy()
        imu_info_Bx6_np = imu_info_Bx6[0].numpy()

        prev_img_RGB_Bx3xHxW = prev_img_RGB_Bx3xHxW[0] * 255.0
        prev_img_RGB_Bx3xHxW = prev_img_RGB_Bx3xHxW.byte()
        prev_img_RGB_Bx3xHxW = prev_img_RGB_Bx3xHxW.permute(2, 1, 0)
        prev_img_RGB_Bx3xHxW = prev_img_RGB_Bx3xHxW.numpy()
        prev_img_RGB_Bx3xHxW_np=prev_img_RGB_Bx3xHxW
        dummy = prev_img_RGB_Bx3xHxW_np.copy()  #Create a new array in memory with a standard C-contiguous layout. 否则会出错！！！！
        prev_img_RGB_Bx3xHxW_np = dummy #calibration.distort_by_calibration(dummy, dst_calib, src_calib)
        prev_img_RGB_Bx3xHxW = torch.tensor(np.transpose(prev_img_RGB_Bx3xHxW_np.astype(np.float32)/255, (2, 1, 0))).unsqueeze(0)
        
        
        #prev_seg_A_Bx1xHxW_np = np.transpose(prev_seg_A_Bx1xHxW[0].repeat(3, 1, 1).numpy(), (2, 1, 0))#.astype(np.uint8)
        prev_seg_A_Bx1xHxW = prev_seg_A_Bx1xHxW[0].repeat(3, 1, 1) * 255.0
        prev_seg_A_Bx1xHxW = prev_seg_A_Bx1xHxW.byte()
        prev_seg_A_Bx1xHxW = prev_seg_A_Bx1xHxW.permute(2, 1, 0)
        prev_seg_A_Bx1xHxW = prev_seg_A_Bx1xHxW.numpy()
        prev_seg_A_Bx1xHxW_np = prev_seg_A_Bx1xHxW
        dummy = prev_seg_A_Bx1xHxW_np.copy()  #Create a new array in memory with a standard C-contiguous layout. 否则会出错！！！！
        prev_seg_A_Bx1xHxW_np = dummy #calibration.distort_by_calibration(dummy, dst_calib, src_calib)
        prev_seg_A_Bx1xHxW = torch.tensor(np.transpose(prev_seg_A_Bx1xHxW_np[:,:,:1].astype(np.float32)/255, (2, 1, 0))).unsqueeze(0)
        
        warped_rgb = warp_rgb_to_new_view(prev_img_RGB_Bx3xHxW_np, prev_depth_tensor_BxHxW_np, prev_imu_info_Bx6_np, imu_info_Bx6_np, K_batched)
        warped_rgb_cache = warped_rgb
        warped_rgb_cache = torch.tensor(np.transpose(warped_rgb_cache.astype(np.float32)/255, (2, 1, 0))).unsqueeze(0)
        warped_rgb = warped_rgb #calibration.distort_by_calibration(warped_rgb, src_calib, dst_calib)
        warped_rgb = torch.tensor(np.transpose(warped_rgb.astype(np.float32)/255, (2, 1, 0))).unsqueeze(0)

        warped_seg = warp_rgb_to_new_view(prev_seg_A_Bx1xHxW_np, prev_depth_tensor_BxHxW_np, prev_imu_info_Bx6_np, imu_info_Bx6_np, K_batched)
        warped_seg_cache = warped_seg
        warped_seg_cache = torch.tensor(np.transpose(warped_seg_cache[:,:,:1].astype(np.float32)/255, (2, 1, 0))).unsqueeze(0)
        warped_seg = warped_seg #calibration.distort_by_calibration(warped_seg, src_calib, dst_calib)
        warped_seg = torch.tensor(np.transpose(warped_seg[:,:,:1].astype(np.float32)/255, (2, 1, 0))).unsqueeze(0)

        if i == 0:
            cached_seg = prev_seg_A_Bx1xHxW * obj_id #torch.zeroslike(prev_seg_A_Bx1xHxW)
            dummy = prev_seg_A_Bx1xHxW_np.copy()
            cached_seg_save = dummy #calibration.distort_by_calibration(dummy, src_calib, dst_calib)
            cached_seg_save = torch.tensor(np.transpose(cached_seg_save[:,:,:1].astype(np.float32)/255, (2, 1, 0))).unsqueeze(0)

            cached_rgb = prev_img_RGB_Bx3xHxW   #校正过，float32 [0,1]
            dummy = prev_img_RGB_Bx3xHxW_np.copy()
            cached_rgb_save = dummy #calibration.distort_by_calibration(dummy, src_calib, dst_calib)
            cached_rgb_save = torch.tensor(np.transpose(cached_rgb_save.astype(np.float32)/255, (2, 1, 0))).unsqueeze(0)

            init_pos = prev_imu_info_Bx6_np
            init_rgb = prev_img_RGB_Bx3xHxW
            init_seg = prev_seg_A_Bx1xHxW
            init_depth = prev_depth_tensor_BxHxW_np
        else:
            coords = F_HW_Bx2.unsqueeze(1)  # 将形状从 (B, 2) 变为 (B, 1, 2)
            # 将归一化坐标转换回像素坐标
            x_pixel = (coords[:, :, 0] * (1408 - 1) + 0.5).long().clamp(0, 1408 - 1)
            y_pixel = (coords[:, :, 1] * (1408 - 1) + 0.5).long().clamp(0, 1408 - 1)
            #print(f'x_pixel {x_pixel}, y_pixel {y_pixel}')
            #print(f'gaze point value {cached_seg[0,0,int(x_pixel.item()),int(y_pixel.item())]}')

            is_valid = cached_seg[0,0,int(x_pixel.item()),int(y_pixel.item())].item() != 0. 
            print('gaze mask valid ', is_valid)
            threshold = 0.0
            gaze_x_right = int((x_pixel+200).clamp(0, 1408 - 1).item())
            gaze_x_left = int((x_pixel-200).clamp(0, 1408 - 1).item())
            gaze_y_right = int((x_pixel+200).clamp(0, 1408 - 1).item())
            gaze_y_left = int((x_pixel-200).clamp(0, 1408 - 1).item())
            diff_mask_per_channel = torch.abs(cached_rgb[0][:,gaze_x_left:gaze_x_right,gaze_y_left:gaze_y_right] - warped_rgb[0][:,gaze_x_left:gaze_x_right,gaze_y_left:gaze_y_right]) > threshold
            different_pixels_mask = torch.any(diff_mask_per_channel, dim=0)
            different_pixels_count = torch.sum(different_pixels_mask).item()
            total_pixels = 1408*1408
            percentage_different_pixels = (different_pixels_count / total_pixels) * 100
            print(f'image difference percentage {percentage_different_pixels}')
            if not is_valid or i % 10 == 0 or percentage_different_pixels > 20:
                init_seg = warped_seg_cache
                init_rgb = warped_rgb_cache
                init_pos = prev_imu_info_Bx6_np
                init_depth = prev_depth_tensor_BxHxW_np
                obj_id += 1
                cached_rgb = warped_rgb
                cached_rgb_save = warped_rgb
                cached_seg = warped_seg
                cached_seg_save = warped_seg
            else:
                cached_rgb = np.transpose(init_rgb[0].numpy()*255, (2, 1, 0)).astype(np.uint8)
                #print('cached_rgb shape ', cached_rgb.shape)
                cached_rgb = warp_rgb_to_new_view(cached_rgb, init_depth, init_pos, imu_info_Bx6_np, K_batched)
                dummy = cached_rgb.copy()
                cached_rgb_save = dummy #calibration.distort_by_calibration(dummy, src_calib, dst_calib)
                cached_rgb = torch.tensor(np.transpose(cached_rgb.astype(np.float32)/255, (2, 1, 0))).unsqueeze(0)
                cached_rgb_save = torch.tensor(np.transpose(cached_rgb_save.astype(np.float32)/255, (2, 1, 0))).unsqueeze(0)
                
                cached_seg = np.transpose(init_seg[0].repeat(3, 1, 1).numpy().astype(np.uint8)*255, (2, 1, 0))
                #print('cached seg shape ', cached_seg.shape)
                cached_seg = warp_rgb_to_new_view(cached_seg, init_depth, init_pos, imu_info_Bx6_np, K_batched)
                dummy = cached_seg.copy()
                cached_seg_save = dummy #calibration.distort_by_calibration(dummy, src_calib, dst_calib)
                cached_seg_save = torch.tensor(np.transpose(cached_seg_save[:,:,:1].astype(np.float32)/255, (2, 1, 0))).unsqueeze(0)
                cached_seg = torch.tensor(np.transpose(cached_seg[:,:,:1].astype(np.float32)/255, (2, 1, 0))).unsqueeze(0)

        
        #cached_seg[0,0,int(x_pixel.item())-10: int(x_pixel.item())+10,int(y_pixel.item())-10:int(y_pixel.item())+10] = 2.
        

        iou = torch.sum(cached_seg*warped_seg).item() / torch.sum((torch.clamp(cached_seg + warped_seg, 0.0, 1.0))).item()
        print('iou ', iou)
        iou_sum += iou
        # if is_valid: # and distance(cached_rgb, img_RGB_Bx3xHxW) < a #之前的物体
        #     cached_seg = cached_seg 
        #     cached_rgb = cached_rgb
        # else:   #新物体
        #     obj_id += 1
        #     #cached_seg = cached_seg + seg_A_Bx1xHxW * obj_id
        #     cached_seg = cached_seg #seg_A_Bx1xHxW
        #     #cached_seg = torch.clamp(cached_seg + seg_A_Bx1xHxW, 0.0, 1.0) #* obj_id
        print('obj_id ', obj_id)
        #print(cached_seg.shape)
        
        first_fid = fid[0].item() if isinstance(fid, torch.Tensor) else fid[0]
        if i % 1 == 0:
            # 获取批次中的第一个数据            
            first_img_rgb = img_RGB_Bx3xHxW[0]  # (C, H, W)
            first_img_rgb_prev = prev_img_RGB_Bx3xHxW[0]  # (C, H, W)
            first_seg_a = seg_A_Bx1xHxW[0]  # (1, H, W)
            first_seg_a_prev = prev_seg_A_Bx1xHxW[0]  # (1, H, W)
            first_warped_rgb = warped_rgb[0]  # (C, H, W)
            first_warped_seg = warped_seg[0]  # (1, H, W)
            first_imu_info = imu_info_Bx6[0]
            first_depth = prev_depth_tensor_BxHxW[0]
            
            # 构建文件名
            img_filename = os.path.join(img_dir, f"{first_fid}_img.png")
            img_prev_filename = os.path.join(img_prev_dir, f"{first_fid}_img_prev.png")
            seg_filename = os.path.join(seg_dir, f"{first_fid}_seg.png")
            seg_prev_filename = os.path.join(seg_prev_dir, f"{first_fid}_seg_prev.png")
            warped_rgb_filename = os.path.join(warped_rgb_dir, f"{first_fid}_warped_rgb.png")
            warped_seg_filename = os.path.join(warped_seg_dir, f"{first_fid}_warped_seg.png")
            imu_filename = os.path.join(imu_dir, f"{first_fid}_imu.pt")
            depth_filename = os.path.join(depth_dir, f"{first_fid}_depth.pt")
            
            # 保存图片
            save_image(first_img_rgb, img_filename)
            save_image(first_img_rgb_prev, img_prev_filename)
            save_image(first_seg_a, seg_filename)
            save_image(first_seg_a_prev, seg_prev_filename)
            save_image(first_warped_rgb, warped_rgb_filename)
            save_image(first_warped_seg, warped_seg_filename)
            torch.save(first_imu_info, imu_filename)
            torch.save(first_depth, depth_filename)
            depth_save_path = os.path.join(depth_dir, f"{first_fid}_depth.png")
            save_depth_img(np.transpose(prev_depth_tensor_BxHxW_np/1000, (1, 0)), depth_save_path)

        first_cacheed_seg = cached_seg_save[0]
        cached_seg_filename = os.path.join(cached_seg_dir, f"{first_fid}_cached_seg.png")
        #print(first_cacheed_seg.shape)
        save_image(first_cacheed_seg, cached_seg_filename)
        #save_segmentation_as_rgb(first_cacheed_seg, output_path=cached_seg_filename, color_map=None)

        first_cacheed_rgb = cached_rgb_save[0]
        cached_rgb_filename = os.path.join(cached_rgb_dir, f"{first_fid}_cached_rgb.png")
        save_image(first_cacheed_rgb, cached_rgb_filename)

        i += 1
        if i > 40:
            print('seg num ', obj_id)
            print('iou_sum ', iou_sum / i)
            break

def profile():
    bs = 1
    dataset = DatasetADT()
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)

    for cache_len in range(1, 2):
        cache = []
        item_durations = []
        i = 0
        hit = 0
        init_iid = -1
        for batch in tqdm(dataloader):
            fid, img_RGB_Bx3xHxW, F_HW_Bx2, seg_A_Bx1xHxW, Y_cls_Bx1, imu_info_Bx6, iid, prev_fid, prev_img_RGB_Bx3xHxW, prev_F_HW_Bx2, prev_seg_A_Bx1xHxW, prev_Y_cls_Bx1, prev_imu_info_Bx6, prev_iid, prev_depth_tensor_BxHxW = batch
            iid = iid[0].item()
            if init_iid != iid:
                item_durations.append(1)
                init_iid = iid
            else:
                item_durations[-1] += 1
            if len(cache) < cache_len and iid not in cache:
                cache.append(iid)
            elif len(cache) == cache_len and iid not in cache:
                cache = cache[1:]
                cache.append(iid)
            else:
                hit += 1
            i += 1
            if i > 800:
                break
        print(f'hit rate when cache len is {cache_len}: {hit/i*100:.2f}%')
        item_durations_filtered = []
        for item in item_durations:
            if item > 3:
                item_durations_filtered.append(item)
        print(f'average gaze duration: {sum(item_durations_filtered) / len(item_durations_filtered):.2f}')

def create_depth_predictor():
    """
    Load depth prediction model from checkpoint.
    Uses default checkpoint path from the command line example.
    
    Returns:
        tuple: (model, device) - The loaded model and the device it's on
    """
    import sys
    from pathlib import Path
    
    # Add depth CNN path to system path
    depth_cnn_path = '/home/external/ORB_SLAM3_VIO/depthCNN'
    if depth_cnn_path not in sys.path:
        sys.path.append(depth_cnn_path)
    
    # Import required modules
    from spatial_patch_encoder_aux import SpatialPatchDepthPredictorWithAux
    
    # Set checkpoint path (default from command line example)
    checkpoint_path = '/home/external/ORB_SLAM3_VIO/depthCNN/checkpoints/spatial_gaze_replication_16x16/checkpoint_best.pth'
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    
    # Load checkpoint to get saved args
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get saved args from checkpoint
    saved_args = checkpoint.get('args', {})
    if hasattr(saved_args, 'depth_patch_size'):
        patch_size = getattr(saved_args, 'depth_patch_size', 16)
        spatial_region_size = getattr(saved_args, 'spatial_region_size', 5)
        encoder_levels = getattr(saved_args, 'encoder_levels', 3)
        base_channels = getattr(saved_args, 'base_channels', 32)
    else:
        # Default values if args not found
        patch_size = 16
        spatial_region_size = 5
        encoder_levels = 3
        base_channels = 32
    
    # Create model with correct architecture (spatial_aux)
    model = SpatialPatchDepthPredictorWithAux(
        image_size=88,  # Default input size
        num_encoder_levels=encoder_levels,
        base_channels=base_channels,
        spatial_region_size=spatial_region_size,
        patch_size=patch_size,
        use_auxiliary_losses=True
    )
    
    # Load model state
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded spatial_aux depth predictor from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, device

def use_depth_predictor(model, device, image_path=None, gaze_x=None, gaze_y=None):
    """
    Predict depth at gaze point using the loaded model.
    Uses default values from command line example if not provided.
    
    Args:
        model: Loaded depth prediction model
        device: Torch device (cuda or cpu)
        image_path: Path to input image (default: ADT test image)
        gaze_x: X coordinate of gaze pixel (default: 1050)
        gaze_y: Y coordinate of gaze pixel (default: 750)
    
    Returns:
        float: Predicted depth at the gaze point in meters
    """
    from PIL import Image
    
    # Use defaults from command line example if not provided
    if image_path is None:
        image_path = '/mnt/ssd_ext/incSeg-data/processed_adt/test/Apartment_release_clean_seq148_M1292/rgb/frame_000450.png'
    if gaze_x is None:
        gaze_x = 1050
    if gaze_y is None:
        gaze_y = 750
    
    # Load image
    if isinstance(image_path, str):
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    elif isinstance(image_path, np.ndarray):
        # Handle numpy array input
        image = image_path
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            image_tensor = torch.from_numpy(image).float()
    elif hasattr(image_path, 'convert'):  # PIL Image
        image_np = np.array(image_path.convert('RGB')).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    else:
        image_tensor = image_path
    
    # Model expects 88x88 context and 44x44 patch
    # Resize full image to 88x88 for context
    context = F.interpolate(
        image_tensor.unsqueeze(0),
        size=(88, 88),
        mode='bilinear',
        align_corners=True
    )
    
    # Create 44x44 patch (for this model, it's just resized version)
    patch = F.interpolate(
        image_tensor.unsqueeze(0),
        size=(44, 44),
        mode='bilinear',
        align_corners=True
    )
    
    # Scale gaze coordinates from 1408x1408 to 88x88
    scaled_x = gaze_x * 88 / 1408
    scaled_y = gaze_y * 88 / 1408
    
    # Prepare inputs
    context = context.to(device)
    patch = patch.to(device)
    gaze_x_tensor = torch.tensor([scaled_x], dtype=torch.float32).to(device)
    gaze_y_tensor = torch.tensor([scaled_y], dtype=torch.float32).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(context, patch, gaze_x_tensor, gaze_y_tensor)
        depth_output = outputs['depth']  # Shape: [1, 16, 16]
        
        # Extract center pixel as the depth at gaze point
        center_y = depth_output.shape[1] // 2  # 8
        center_x = depth_output.shape[2] // 2  # 8
        depth = depth_output[0, center_y, center_x].item()
    
    return depth

if __name__ == '__main__':
    #try_transform()
    #run_dataset()
    #profile()
    #try_np_transform()
    #try_torch_transform()
    
    # Load the depth prediction model
    model, device = create_depth_predictor()
    
    # Use the model with default values from command line example
    predicted_depth = use_depth_predictor(model, device)
    
    # Print the result
    print(f"Predicted depth at gaze point (1050, 750): {predicted_depth:.3f} meters")
