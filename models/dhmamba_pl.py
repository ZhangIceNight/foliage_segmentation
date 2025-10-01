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

        # keep original state
        orig = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)  # disable deterministic
        loss = self.loss_fn(logits, labels.squeeze())
        torch.use_deterministic_algorithms(orig)  # restore original state

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def save_predictions(self, file_names, preds, points, labels, save_root="./Results/visualization_results/"):
        """
        save predictions to txt files.
        file_names: npz files loaded from dataloader
        preds: from model (tensor), automatically convert to numpy and save.

        Example:
        file_name: ./data/Larch/tiles_filtered_fps/tte1.npz
        save to: ./Results/visualization_results/Larch/tte1.txt
        """
        # preds to list[np.ndarray]
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
            points = points.cpu().numpy()
            labels = labels.cpu().numpy()

        for i in range(len(preds)):
            file_name, pred, point, label = file_names[i], preds[i], points[i], labels[i]

            # process each file in the batch
            base_name = os.path.basename(file_name)         # tte1.npz
            name_no_ext_pred = os.path.splitext(base_name)[0] + "_pred"   # tte1_pred
            name_no_ext_point = os.path.splitext(base_name)[0] + "_point"   # tte1_point
            name_no_ext_label = os.path.splitext(base_name)[0] + "_label"   # tte1_label
            grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(file_name)))  # Larch

            # construct save directory
            save_dir = os.path.join(save_root, grandparent_dir)
            os.makedirs(save_dir, exist_ok=True)
            save_path_pred = os.path.join(save_dir, name_no_ext_pred + ".txt")
            save_path_point = os.path.join(save_dir, name_no_ext_point + ".txt")
            save_path_label = os.path.join(save_dir, name_no_ext_label + ".txt")

            # save predictions
            np.savetxt(save_path_pred, pred.astype(int), fmt="%d")
            np.savetxt(save_path_point, point.astype(float), fmt="%f")
            np.savetxt(save_path_label, label.astype(int), fmt="%d")
            print(f"Successfully saved to: {save_path_pred}, {save_path_point}, {save_path_label}")

    def validation_step(self, batch, batch_idx):
        points, labels, _, file_names = batch
        logits = self.model(points) # [B, N_classes, N]
        preds_save = logits.argmax(dim=1)   # (B, N)
        # file_names is a list with length of B，preds: [B, N] tensor
        self.save_predictions(file_names, preds_save, points, labels)

        # keep original state
        orig = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)  # disable deterministic
        loss = self.loss_fn(logits, labels.squeeze())
        torch.use_deterministic_algorithms(orig)  # restore original state
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        miou = self.calculate_iou(preds, labels, self.model_hparams['num_classes'])

        # save validation loss & accuracy
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
                ious.append(torch.tensor(1.0, device=pred.device)) 
            else:
                ious.append(intersection / union)

        return torch.mean(torch.stack(ious))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.opt_hparams["learning_rate"], weight_decay=self.opt_hparams["weight_decay"])
 
        total_epochs = self.opt_hparams["max_epochs"]  
        warmup_epochs = self.opt_hparams["warmup_epochs"]

        scheduler_warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
    
        scheduler_cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,  
            eta_min=self.opt_hparams["eta_min"]
        )
    
        combined_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_epochs]  
        )
 
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': combined_scheduler,
                'interval': 'epoch', 
                'frequency': 1
            }
        }
   

 
