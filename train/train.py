import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from omegaconf import OmegaConf

from network.cerberus_wrapper import CerberusWrapper
from dataset.slakh2100 import Slakh2100
from utils.debug import set_debug_mode

set_debug_mode()
config = OmegaConf.load("../configs/config.yaml")
config.merge_with_cli()

pl.trainer.seed_everything(config.seed)

net = CerberusWrapper(config)

ckpt_callback = ModelCheckpoint(
    monitor="valid_total_loss",
    filename="model-{epoch:03d}-{valid_total_loss:.4f}",
    save_top_k=1,
    mode='min',
    save_last=True
    )

logger = TensorBoardLogger(save_dir="lightning_logs", 
                           name=config.experiment_name, 
                           default_hp_metric=False)

lr_monitor = LearningRateMonitor(logging_interval='epoch')

trainer = pl.Trainer(
    gpus=config.num_gpu,
    min_epochs=1,
    max_epochs=config.num_epochs,
    checkpoint_callback=ckpt_callback,
    check_val_every_n_epoch=config.check_val_every_n_epoch,
    gradient_clip_val=config.gradient_clip_val,
    auto_lr_find=config.find_lr,
    logger=logger,
    callbacks=[lr_monitor],
    weights_summary='full',
    profiler='simple',
)

if config.find_lr:
    trainer.tune(net)

trainer.fit(net)

trainer.test(model=net)
