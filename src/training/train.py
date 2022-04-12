from argparse import ArgumentParser
import os

import pytorch_lightning as pl
import torch
import wandb

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from data import WeatherBenchSuperresolutionDataModule
from models import LitSuperresolutionModelWrapper


COARSE_SUB_DIR = "5625/temp_5625_processed.zarr"  # 5.625 degrees
FINE_SUB_DIR = "1406/temp_1406_processed.zarr"  # 1.402 degrees


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser("Superresolution CNN Trainer")
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-d', '--data_dir', default="./data/processed/temp/", type=str)
    parser.add_argument('-m', '--model', default="LM", type=str)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # setup
    # ------------
    pl.seed_everything(1729)

    model = LitSuperresolutionModelWrapper(model_keyword=args.model)

    data_module = WeatherBenchSuperresolutionDataModule(
        coarse_dir = os.path.join(args.data_dir, COARSE_SUB_DIR),
        fine_dir = os.path.join(args.data_dir, FINE_SUB_DIR),
        batch_size = args.batch_size
    )

    wandb_exp = wandb.init(project="cnn", entity="6759-proj")

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(
        gpus=1, accelerator='gpu', max_epochs=1000,
        callbacks=[
            EarlyStopping(monitor="Validation Loss", mode="min", patience=9),
            ModelCheckpoint(
                monitor="Validation Loss", mode="min",
                dirpath=f"./artifacts/checkpoints/{wandb_exp.name}/",
                filename=f"{wandb_exp.name}_best_val_loss_" + "{epoch:04d}"
            )
        ],
        logger = WandbLogger(save_dir="./logs/")
    )

    trainer.fit(model, data_module)

    # ------------
    # testing
    # ------------
    # trainer.test(test_dataloaders=test_loader)




if __name__ == '__main__':
    cli_main()
