import pytorch_lightning as pl
import torch
from torch import optim, nn
from .dhmamba import DHMamba
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

class DHMamba_pl(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model_hparams = config.model
        self.opt_hparams = config.optimizer
        self.model = DHMamba(num_classes=self.model_hparams['num_classes'])
        self.loss_fn = nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        points, labels = batch
        logits = self.model(points)
        loss = self.loss_fn(logits, labels.squeeze())
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        points, labels = batch
        logits = self.model(points)
        loss = self.loss_fn(logits, labels.squeeze())
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        iou = self.calculate_iou(preds, labels, self.model_hparams['num_classes'])

        # 记录验证损失 & 准确率
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", accuracy, prog_bar=True, logger=True)
        self.log("val_iou", iou, prog_bar=True, logger=True)

        return {
            "val_loss": loss,
            "val_acc": accuracy,
            "val_iou": iou
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
   

 
