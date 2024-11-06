import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, 6, [64, 64, 128], False)
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256], False)
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], True)
        
        self.fp3 = PointNetFeaturePropagation(1280, [256, 256])
        self.fp2 = PointNetFeaturePropagation(384, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128 + 16 + 6, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        # Set Abstraction layers
        B,C,N = xyz.shape

        l0_points = xyz
        l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        # FC layers
        feat =  F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))