from argparse import ArgumentParser
import os

import torch
import pytorch_lightning as pl

from data import WeatherBenchSuperresolutionDataModule
from models import LitSuperresolutionModelWrapper


COARSE_SUB_DIR = "5625/temp_5625_processed.zarr"  # 5.625 degrees
FINE_SUB_DIR = "1406/temp_1406_processed.zarr"  # 1.402 degrees


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser("Superresolution CNN Trainer")
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('-d', '--data_dir', default="./data/processed/temp/", type=str)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # setup
    # ------------
    pl.seed_everything(1729)

    model = LitSuperresolutionModelWrapper(model_keyword="LM")

    data_module = WeatherBenchSuperresolutionDataModule(
        coarse_dir = os.path.join(args.data_dir, COARSE_SUB_DIR),
        fine_dir = os.path.join(args.data_dir, FINE_SUB_DIR),
        batch_size = args.batch_size
    )

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, data_module)

    # ------------
    # testing
    # ------------
    # trainer.test(test_dataloaders=test_loader)




if __name__ == '__main__':
    cli_main()
