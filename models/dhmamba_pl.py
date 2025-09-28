import pytorch_lightning as pl
import torch
from torch import optim, nn
from .dhmamba import DHMamba
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import numpy as np
import os
class DHMamba_pl(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model_hparams = config.model
        self.opt_hparams = config.optimizer
        self.model = DHMamba(num_classes=int(self.model_hparams['num_classes']), HGNeighbors=int(self.model_hparams['HGNeighbors']))
        self.loss_fn = nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        points, labels, _ = batch
        logits = self.model(points)
        # print("train logits:", logits.shape, "train labels:", labels.shape)
        # 保存原状态
        orig = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)  # 关闭 deterministic
        loss = self.loss_fn(logits, labels.squeeze())
        torch.use_deterministic_algorithms(orig)  # 恢复原状态

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def save_predictions(self, file_names, preds, save_root="/home/wjzhang/workspace/results/DHMamba"):
        """
        保存预测结果到 txt 文件。
        file_name 来自 dataloader (通常是 .npz)，
        pred 来自模型 (tensor)，自动转 numpy 并保存。

        Example:
        file_name = /home/wjzhang/data/Larch/tte1.npz
        保存到   /home/wjzhang/workspace/results/DHMamba/Larch/tte1.txt
        """
        # 保证 preds 转成 list[np.ndarray]
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(preds, np.ndarray):
            preds = [preds]  # 单个样本时

        for file_name, pred in zip(file_names, preds):
            print(f"正在保存预测结果: {file_name} ...")
            print(f"pred shape: {pred.shape}, unique labels: {np.unique(pred)}")
            # 处理 batch 内每个文件
            base_name = os.path.basename(file_name)         # tte1.npz
            name_no_ext = os.path.splitext(base_name)[0]    # tte1
            parent_dir = os.path.basename(os.path.dirname(file_name))  # Larch

            # 构造保存目录
            save_dir = os.path.join(save_root, parent_dir)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, name_no_ext + ".txt")

            # 保存预测结果
            np.savetxt(save_path, pred.astype(int), fmt="%d")
            print(f"已成功保存到: {save_path}")

    def validation_step(self, batch, batch_idx):
        points, labels, _, file_names = batch
        logits = self.model(points) # [B, N_classes, N]
        # print("val logits:", logits.shape, "val labels:", labels.shape)

        preds_save = logits.argmax(dim=1)   # (B, N)
        # file_names 是长度为 B 的列表，preds 是 (B, N) tensor
        self.save_predictions(file_names, preds_save)
        
        # 保存原状态
        orig = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)  # 关闭 deterministic
        loss = self.loss_fn(logits, labels.squeeze())
        torch.use_deterministic_algorithms(orig)  # 恢复原状态
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        miou = self.calculate_iou(preds, labels, self.model_hparams['num_classes'])

        # 记录验证损失 & 准确率
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", accuracy, prog_bar=True, logger=True)
        self.log("val_mIoU", miou, prog_bar=True, logger=True)

        return {
            "val_loss": loss,
            "val_acc": accuracy,
            "val_mIoU": miou
        }

    @staticmethod
    def calculate_iou(pred, target, num_classes):
        pred = pred.view(-1)
        target = target.view(-1)

        ious = []
        for cls in range(num_classes):
            pred_inds = (pred == cls)
            target_inds = (target == cls)

            intersection = (pred_inds & target_inds).sum().float()
            union = pred_inds.sum().float() + target_inds.sum().float() - intersection

            if union == 0:
                ious.append(torch.tensor(1.0, device=pred.device))  # 保持在同一 device
            else:
                ious.append(intersection / union)

        return torch.mean(torch.stack(ious))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.opt_hparams["learning_rate"], weight_decay=self.opt_hparams["weight_decay"])
 
        total_epochs = self.opt_hparams["max_epochs"]  # 例如：50
        warmup_epochs = self.opt_hparams["warmup_epochs"]

        scheduler_warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
    
        # Cosine 退火阶段：从 max lr 衰减到接近 0
        scheduler_cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,  
            eta_min=self.opt_hparams["eta_min"]
        )
    
        # 合并两个调度器为一个阶段式调度器
        combined_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_epochs]  # 第 5 个 epoch 结束后切换到 cosine
        )
 
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': combined_scheduler,
                'interval': 'epoch', 
                'frequency': 1
            }
        }
        # return optim.Adam(self.parameters(), lr=self.opt_hparams["learning_rate"])
   

 
