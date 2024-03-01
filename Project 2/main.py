
import argparse
import sys
import os
from os.path import dirname, realpath

sys.path.append(dirname(dirname(realpath(__file__))))
from model import *
from utils import *
from torch import nn
import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import *
import pandas as pd

from lightning.pytorch.cli import LightningArgumentParser
import lightning.pytorch as pl

NAME_TO_DATASET_CLASS = {
    "ist": IST,
    "ihdp": IHDP
}
METHOD = {
    -1: 'TARNet',
    0: 'CFRWass',
    1: 'CFRMMD'
}

def add_main_args(parser: LightningArgumentParser) -> LightningArgumentParser:

    parser.add_argument(
        "--project_name",
        default="cornerstone",
        help="Name of project for wandb"
    )

    parser.add_argument(
        "--dataset_name",
        default="ihdp",
        help="Name of dataset"
    )

    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Path to checkpoint to load from. If None, init from scratch."
    )

    parser.add_argument(
        "--train",
        default=False,
        action="store_true",
        help="Whether to train the model."
    )

    parser.add_argument(
        "--wandb_offline",
        default=False,
        help="Whether to use wandb online."
    )

    parser.add_argument(
        "--monitor_key",
        default="val_loss",
        help="Name of metric to use for checkpointing. (e.g. val_loss, val_acc)"
    )

    return parser

def parse_args() -> argparse.Namespace:
    parser = LightningArgumentParser(default_config_files=['config.yaml'])
    parser.add_lightning_class_args(pl.Trainer, nested_key="trainer")
    parser.add_lightning_class_args(CFRNet, nested_key='model')
    for dataset_name, data_class in NAME_TO_DATASET_CLASS.items():
        parser.add_lightning_class_args(data_class, nested_key=dataset_name)
    parser = add_main_args(parser)
    args = parser.parse_args()
    return args

from sklearn.model_selection import train_test_split

def main(args):
    print(args)

    dm = NAME_TO_DATASET_CLASS[args.dataset_name](**vars(args[args.dataset_name]))

    if args.checkpoint_path is not None:
        model = CFRNet.load_from_checkpoint(args.checkpoint_path, **vars(args.model))
    else:
        model = CFRNet(**vars(args.model))

    logger = pl.loggers.WandbLogger(project=args.project_name)
    args.trainer.accelerator = 'auto'
    args.trainer.logger = logger
    args.trainer.precision = "bf16-mixed" ## This mixed precision training is highly recommended

    args.trainer.callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor=args.monitor_key,
            mode='min' if "loss" in args.monitor_key else "max",
            save_last=True
        )]

    trainer = pl.Trainer(**vars(args.trainer))
    if args.train:
        print("Training model")
        trainer.fit(model, dm)
    
    print("Best model checkpoint path: ", trainer.checkpoint_callback.best_model_path)

    print("Evaluating model on validation set")
    trainer.validate(model, dm)

    print("Evaluating model on test set")
    trainer.test(model, dm)
    plot_TSNE(model, args.dataset_name, dm.X, dm.T, METHOD[args.model.loss_mode])
    if args.dataset_name == 'ihdp':
        get_ITE_CFRNet(model, dm.X, dm.T, dm.ihdp_data['ITE'].to_numpy())
    else:
        get_ITE_CFRNet(model, dm.X, dm.T)

    # ITE, correct_predicted_probability = get_ITE_CFRNet(model, dm.test.tensors[0], dm.test.tensors[2], best_treatment=None)

if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args)