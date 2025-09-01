import os
import torch
import numpy as np
from tqdm import tqdm

from datasets.pls_dataset import PLSDataset
from models import pointnet  # 假设这里有 pointnet_seg 模型

class SegmentationTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')

        # 构建 train / val 数据集
        self.train_dataset = PLSDataset(
            root=cfg['data_root'],
            split='train',
            npoints=cfg['num_points'],
            use_uniform_sample=cfg['use_uniform_sample'],
            use_normals=cfg['use_normals'],
            task='seg'   # 加一个参数标识是分割任务
        )
        self.val_dataset = PLSDataset(
            root=cfg['data_root'],
            split='val',
            npoints=cfg['num_points'],
            use_uniform_sample=cfg['use_uniform_sample'],
            use_normals=cfg['use_normals'],
            task='seg'
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

        self.num_classes = self.train_dataset.num_seg_classes
        self.model = pointnet.get_seg_model(self.num_classes, normal_channel=cfg['use_normals']).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg['learning_rate'],
            weight_decay=cfg['weight_decay']
        )

        self.best_mIoU = 0.0
        os.makedirs(os.path.dirname(cfg['save_path']), exist_ok=True)

    def train(self):
        for epoch in range(self.cfg['num_epochs']):
            self.model.train()
            train_losses, train_ious = [], []

            for points, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg['num_epochs']}"):
                points, labels = points.to(self.device), labels.to(self.device)  # [B, N, 3], [B, N]
                points = points.transpose(2, 1)  # [B, 3, N]

                self.optimizer.zero_grad()
                preds, _ = self.model(points)   # preds: [B, num_classes, N]
                preds = preds.transpose(2, 1).contiguous()  # [B, N, num_classes]

                loss = self.criterion(preds.view(-1, self.num_classes), labels.view(-1))
                loss.backward()
                self.optimizer.step()

                # 计算 IoU
                pred_labels = preds.max(dim=2)[1]  # [B, N]
                iou = self.calculate_iou(pred_labels, labels, self.num_classes)

                train_losses.append(loss.item())
                train_ious.append(iou)

            avg_loss = np.mean(train_losses)
            avg_iou = np.mean(train_ious)
            print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}, Train mIoU: {avg_iou:.4f}")

            self.evaluate(epoch)

    def evaluate(self, epoch):
        self.model.eval()
        val_ious = []
        val_losses = []

        with torch.no_grad():
            for points, labels in self.val_loader:
                points, labels = points.to(self.device), labels.to(self.device)
                points = points.transpose(2, 1)
                preds, _ = self.model(points)
                preds = preds.transpose(2, 1).contiguous()

                loss = self.criterion(preds.view(-1, self.num_classes), labels.view(-1))

                pred_labels = preds.max(dim=2)[1]
                iou = self.calculate_iou(pred_labels, labels, self.num_classes)

                val_losses.append(loss.item())
                val_ious.append(iou)

        avg_loss = np.mean(val_losses)
        avg_iou = np.mean(val_ious)
        print(f"[Epoch {epoch+1}] Val Loss: {avg_loss:.4f}, Val mIoU: {avg_iou:.4f}")

        if avg_iou > self.best_mIoU:
            self.best_mIoU = avg_iou
            torch.save(self.model.state_dict(), self.cfg['save_path'])
            print(f"[✓] Saved best model with mIoU {avg_iou:.4f}")

    
