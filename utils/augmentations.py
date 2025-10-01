# utils/augmentations.py

import numpy as np

def pc_normalize(pc):
    """normalize point cloud: center + scale to unit sphere"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def random_sample_dropout(pc, max_dropout_ratio=0.875):
    """randomly drop a portion of the point cloud"""
    dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    if len(drop_idx) == 0:
        return pc
    pc[drop_idx] = pc[0]
    return pc

def random_scale_point_cloud(pc, scale_low=0.8, scale_high=1.25):
    """randomly scale point cloud"""
    scale = np.random.uniform(scale_low, scale_high)
    pc *= scale
    return pc

def random_shift_point_cloud(pc, shift_ratio=0.1):
    """random shift point cloud"""
    shift = np.random.uniform(-shift_ratio, shift_ratio, 3)
    pc += shift
    return pc

def random_rotate_point_cloud_y_axis(pc):
    """random rotate point cloud around y axis"""
    angle = np.random.uniform() * 2 * np.pi
    cos, sin = np.cos(angle), np.sin(angle)
    R = np.array([[cos, 0, sin],
                  [0, 1, 0],
                  [-sin, 0, cos]])
    pc = np.dot(pc, R)
    return pc
