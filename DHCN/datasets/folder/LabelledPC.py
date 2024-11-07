import torch.utils.data as data
import os
import scipy.io as scio
import torch
import numpy as np
from PIL import Image, ImagePath
import pandas as pd


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        # 将图像调整为224x224
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        return img


class LabelledPC(data.Dataset):

    def __init__(self, root, transform, istrain, config):
        self.istrain = istrain
        self.config = config
        self.data = []
        self.img_path = root + "distorted2D"
        self.pc_path = root + "LabelledPC_6patch_2048"
        # 根据istrain选择对应的划分文件
        split_file = os.path.join(root, 'trainval.txt' if istrain else 'test.txt')
        
        # 读取划分文件
        with open(split_file, 'r') as f:
            model_names = [line.strip() for line in f.readlines()]
            # 移除.npy后缀
            model_names = [name.split('.npy')[0] for name in model_names]

        use_number = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]

        for model_name in model_names:
            file_list = []
            # 收集每个视角的投影图像
            for j in use_number:
                file_name = f"{model_name}_{j}.png"
                file_path = os.path.join(self.img_path, file_name)
                file_list.append(file_path)

            # patch文件路径
            ply_name = f"patch_{model_name}.npy"  # 注意这里添加了patch_前缀
            ply_path = os.path.join(self.pc_path, ply_name)

            self.data.append((file_list, ply_path))

        self.transform = transform
        self.patch_length_read = 6
        self.npoint = 2048
        print("load dataset num:", len(self.data))

    def __getitem__(self, index):

        file_list, ply_path = self.data[index]
        imgs = None
        for i in file_list:
            if imgs is None:
                imgs = self.transform(pil_loader(i)).unsqueeze(0)
            else:
                file = self.transform(pil_loader(i)).unsqueeze(0)
                imgs = torch.cat((imgs, file), dim=0)

        selected_patches = torch.zeros([self.patch_length_read, 3, self.npoint])
        labels = torch.zeros([self.patch_length_read, 1, self.npoint])
        points = list(np.load(ply_path))

        for i in range(self.patch_length_read):
            selected_patches[i] = torch.from_numpy(points[i][:, :3]).transpose(0, 1)
            labels[i] = torch.from_numpy(points[i][:, -1])
        return imgs, labels, selected_patches

    def __len__(self):
        length = len(self.data)
        return length
