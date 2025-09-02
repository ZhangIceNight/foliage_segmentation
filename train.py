import comet_ml

import logging
import os

from omegaconf import DictConfig, OmegaConf
import hydra

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from models import build_model
from datasets.ForestSemantic_Difficult_Dataset import ForestSemantic_Difficult_DataModule
from utils.logger_utils import setup_logger, create_experiment_dir
from utils.seed_utils import seed_everything


@hydra.main(config_path="configs", config_name="PLU_AUT_pointnet_lr1e-3_bs32", version_base=None)
def train(config: DictConfig):
    fold = config.data.fold_idx
    exp_dir, ckpt_dir, comet_dir, log_dir = create_experiment_dir(
        root_dir=config.trainer.default_root_dir, 
        dataset_type=config.data.dataset_type,
        model_dir=config.model.model_type,
        fold_idx=fold
    )
    
    # 初始化 logger
    log_file = os.path.join(log_dir, "training.log")
    logger = setup_logger(log_file)
    logger.info(f"===== Starting Fold {fold} =====")

    # 保存当前 config 到实验目录
    OmegaConf.save(config, os.path.join(exp_dir, "config.yaml"))
    print(OmegaConf.to_yaml(config))

    comet_logger = CometLogger(
        project_name=config.comet.get("project"),
        experiment_name=config.comet.get("name"), 
        # offline_directory=comet_dir,
        workspace="zwjnefu"
    )
    comet_logger.experiment.add_tag(f"fold_{fold}")
    comet_logger.experiment.log_parameters({"fold_idx": fold})
    print("CometLogger project:", comet_logger.experiment.project_name)
    print("CometLogger workspace:", comet_logger.experiment.workspace)
    print("CometLogger experiment name:", comet_logger.experiment.get_name())

    exit(0)
    # Setup Dataset Module
    config.data.fold_idx = fold
    data_module = ForestSemantic_Difficult_DataModule(**config.data)
    data_module.setup()

    # Setup Model
    model = build_model(config)

    # Setup Callbacks
    best_checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{config.model.model_type}-{epoch:02d}-val_mIoU{val_mIoU:.2f}",
        monitor="val_mIoU",
        save_top_k=1,
        mode="max",
        auto_insert_metric_name=False
    )

    latest_checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{config.model.model_type}-latest",
        save_top_k=1,
        every_n_epochs=5,
        save_last=True,
    )

    trainer = pl.Trainer(
        logger=comet_logger,
        callbacks=[best_checkpoint_cb, latest_checkpoint_cb],        
        **config['trainer']
    )

    if config.model.get("resume"):
        print(f"Resuming from checkpoint: {config.model.resume}")
        trainer.fit(model, datamodule=data_module, ckpt_path=config.model.resume)
    else:
        print("Training from scratch")
        trainer.fit(model, datamodule=data_module)
    
    best_val_mIoU = best_checkpoint_cb.best_model_score.item()
    comet_logger.experiment.log_metric("best_val_mIoU", best_val_mIoU)
    logger.info(f"Fold {fold} best validation mIoU: {best_val_mIoU:.4f}")
    logger.info(f"===== End Fold {fold} =====")


if __name__ == "__main__":
    seed_everything(42)
    train()
