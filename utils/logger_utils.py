import datetime
import shutil

import logging
import os
import sys

class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass


def setup_logger(log_file='training.log'):
    logging.shutdown()  # clean old handlers
    logging.getLogger().handlers.clear()  # clean cached handlers
    # create directory
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
 
    # create logger
    logger = logging.getLogger('PointCloudTraining')
    logger.setLevel(logging.INFO)
 
    # avoid repeatly add handler
    if not logger.handlers:
        # file handler：log to file
        file_handler = logging.FileHandler(log_file, mode='w')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # terminal handler：log to stderr (so that lightning's progress bar is not affected)
        console_handler = logging.StreamHandler(sys.stderr)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
 
    return logger





def create_experiment_dir(root_dir="Results", dataset_type=None, model_dir=None, fold_idx=0):
    # create timestamp directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    exp_dir = os.path.join(root_dir, dataset_type, model_dir, f"fold_{fold_idx}", f"exp_{timestamp}")

    # create subdirectories
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    comet_dir = os.path.join(exp_dir, "comet_logs")
    log_dir = os.path.join(exp_dir, "logs")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(comet_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    return exp_dir, ckpt_dir, comet_dir, log_dir
