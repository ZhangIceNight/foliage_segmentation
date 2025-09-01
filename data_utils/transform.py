import numpy as np

class Compose:
    """顺序组合多个 transform"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, xyz):
        for t in self.transforms:
            xyz = t(xyz)
        return xyz


class RandomRotateZ:
    """绕 Z 轴随机旋转"""
    def __call__(self, xyz):
        theta = np.random.uniform(0, 2 * np.pi)
        rot_mat = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ], dtype=np.float32)
        return xyz @ rot_mat.T


class RandomScale:
    """随机缩放"""
    def __init__(self, scale_low=0.8, scale_high=1.2):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, xyz):
        scale = np.random.uniform(self.scale_low, self.scale_high)
        return xyz * scale


class RandomShift:
    """随机平移"""
    def __init__(self, shift_range=0.1):
        self.shift_range = shift_range

    def __call__(self, xyz):
        shift = np.random.uniform(-self.shift_range, self.shift_range, size=(1, 3))
        return xyz + shift


class RandomJitter:
    """加噪声"""
    def __init__(self, sigma=0.005, clip=0.02):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, xyz):
        noise = np.clip(self.sigma * np.random.randn(*xyz.shape), -self.clip, self.clip)
        return xyz + noise.astype(np.float32)
