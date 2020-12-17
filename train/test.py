import sys
import os
import glob

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from omegaconf import OmegaConf

from network.cerberus_wrapper import CerberusWrapper
from dataset.slakh2100 import Slakh2100
from utils.debug import set_debug_mode
from metrics.transcript_metric import TrMetrics

version = sys.argv[1] # "lightning_logs/experiment_name/version_0/"
print(version, sys.argv)
checkpoint = glob.glob(version +  "/checkpoints/*.ckpt")
if len(checkpoint) == 0:
    print("No checkpoint at", os.path.join(version, "/checkpoints") + "/*.ckpt")
    exit(0)
checkpoint = checkpoint[0]

config_path = os.path.join(version, "hparams.yaml")
config = OmegaConf.load(config_path)
pl.trainer.seed_everything(config.seed)

net = CerberusWrapper.load_from_checkpoint(checkpoint)

set_debug_mode()
ckpt_callback = ModelCheckpoint(
    monitor="valid_total_loss",
    filename="model-{epoch:03d}-{valid_total_loss:.4f}",
    save_top_k=1,
    mode='min',
    )

logger = TensorBoardLogger(save_dir="lightning_logs", 
                           name=config.experiment_name, 
                           version=version[version.find("version"):],
                           default_hp_metric=False)

lr_monitor = LearningRateMonitor(logging_interval='epoch')

trainer = pl.Trainer(
    gpus=config.num_gpu,
    min_epochs=1,
    max_epochs=config.num_epochs,
    checkpoint_callback=ckpt_callback,
    # val_check_interval=config.val_check_interval,
    check_val_every_n_epoch=config.check_val_every_n_epoch,
    gradient_clip_val=config.gradient_clip_val,
    auto_lr_find=config.find_lr,
    logger=logger,
    callbacks=[lr_monitor],
    weights_summary='full',
    profiler='simple',
)

trainer.test(model = net)