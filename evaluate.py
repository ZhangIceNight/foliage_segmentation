import comet_ml
import logging
import os

from omegaconf import DictConfig, OmegaConf
import hydra

import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

from models import build_model
from datasets.ForestSemantic_Difficult_Dataset import ForestSemantic_Difficult_DataModule
from utils.logger_utils import setup_logger, create_experiment_dir
from utils.seed_utils import seed_everything


@hydra.main(config_path="configs", config_name="PLU_AUT_pointnet_lr1e-3_bs32", version_base=None)
def evaluate(config: DictConfig):
    fold = config.data.fold_idx
    exp_dir, ckpt_dir, comet_dir, log_dir = create_experiment_dir(
        root_dir=config.trainer.default_root_dir, 
        dataset_type=config.data.dataset_type,
        model_dir=config.model.model_type,
        fold_idx=fold
    )
    
    # 初始化 logger
    log_file = os.path.join(log_dir, "evaluate.log")
    logger = setup_logger(log_file)
    logger.info(f"===== Starting Evaluation Fold {fold} =====")

    # 保存 config
    OmegaConf.save(config, os.path.join(exp_dir, "config_eval.yaml"))
    print(OmegaConf.to_yaml(config))

    comet_logger = CometLogger(
        project_name=config.comet.get("project"),
        experiment_name=config.comet.get("name") + "_eval", 
        workspace="zwjnefu"
    )
    comet_logger.experiment.add_tag(f"fold_{fold}")
    comet_logger.experiment.log_parameters({"fold_idx": fold})

    # Dataset
    data_module = ForestSemantic_Difficult_DataModule(**config.data)
    data_module.setup()
    val_loader = data_module.val_dataloader()
    # Model
    if not config.model.get("resume"):
        raise ValueError("You must specify a checkpoint in config.model.resume for evaluation")
    ckpt_path = config.model.resume
    logger.info(f"Loading checkpoint from {ckpt_path}")
    model = build_model(config)
    # Trainer
    trainer = pl.Trainer(
        logger=comet_logger,
        **config['trainer']
    )

    # Run test (or validation)
    trainer.validate(model, datamodule=data_module, ckpt_path=config.model.resume)
    
    # logger.info(f"Evaluation results: {results}")
    # comet_logger.experiment.log_metrics({f"eval_{k}": v for k, v in results[0].items()})

    logger.info(f"===== End Evaluation Fold {fold} =====")


if __name__ == "__main__":
    seed_everything(42)
    evaluate()
