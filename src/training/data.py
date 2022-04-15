""" A PyTorch Lightning data module wrapping paired coarse-grid and fine-grid weather
        observations from WeatherBench
"""

import os

import pytorch_lightning as pl
import torch

import numpy as np
import xarray as xr

from torch.utils.data import DataLoader, TensorDataset

from typing import List


CPUS_AVAILABLE = os.cpu_count() // 2

class WeatherBenchSuperresolutionDataModule(pl.LightningDataModule):
    """ A PyTorch Lightning data module wrapping paired coarse-grid and fine-grid weather
        observations from WeatherBench.

        Assumes highly-processed data can fit in memory.
    """

    def __init__(self, coarse_dir: os.PathLike, fine_dir: os.PathLike, batch_size: int):
        """
        Arguments:
          * coarse_dir: the path to the coarse-grid data
          * fine_dir: the path to the fine-grid data
        """
        super().__init__()

        self.coarse = xr.open_zarr(coarse_dir).to_dataframe().astype(float)
        self.fine = xr.open_zarr(fine_dir).to_dataframe().astype(float)
        self.batch_size = batch_size

        if not (self.coarse.index.levels[0] == self.fine.index.levels[0]).all():
            raise ValueError("coarse-grid dates do not match fine-grid dates")

    def setup(self, stage):
        """ Determines transformation, splits, etc. """
        # demean
        self.daily_means = self.coarse.groupby("date").mean()
        self.daily_std = self.coarse.groupby("date").std()
        coarse_standardized = (self.coarse - self.daily_means) / self.daily_std
        fine_standardized = (self.fine - self.daily_means) / self.daily_std  # can't assume access to fine-grid means!

        # reshape to gridded tensors
        coarse_grid_dims = [len(dim) for dim in coarse_standardized.index.levels]  # (d, h, w)
        coarse_standardized_arr = torch.Tensor(coarse_standardized.values.reshape(coarse_grid_dims)).unsqueeze(dim=1)  # pad with channel dim

        fine_grid_dims = [len(dim) for dim in fine_standardized.index.levels]  # (d, h, w)
        fine_standardized_arr = torch.Tensor(fine_standardized.values.reshape(fine_grid_dims))

        # train / validation / test split (70/20/10)
        num_dates = coarse_standardized_arr.shape[0]
        range_splits = [int(pct*num_dates) for pct in [.7, .2, .1]]

        c_train, c_valid, c_test = torch.split(coarse_standardized_arr, range_splits)
        f_train, f_valid, f_test = torch.split(fine_standardized_arr, range_splits)

        self.train = TensorDataset(c_train, f_train)
        self.validation = TensorDataset(c_valid, f_valid)
        self.test = TensorDataset(c_test, f_test)

        # store date metadata for future use
        self.daily_means.columns = ["mean"]
        self.daily_std.columns = ["std"]
        self.split_date_ranges = self.split_indices(self.coarse, range_splits)

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=1
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation, batch_size=self.batch_size, num_workers=1
        )

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=1
        )

    @staticmethod
    def split_indices(pdf: 'DataFrame', range_splits: List[int]):
        """ Returns training, validation, and test set index ranges.

            Arguments:
             * pdf: a dataframe where length of the level 0 index equals sum(range_splits)
             * range_splits: a list of 3 integers corresponding to size of train/valid/test
        """
        ind0 = pdf.index.levels[0]

        if sum(range_splits) != len(ind0):
            raise ValueError(
                f"length of first index ({len(ind0)}) should equal sum of range_splits {sum(range_splits)}"
            )

        last_ind_ilocs = np.cumsum(range_splits) - 1
        last_inds = ind0[last_ind_ilocs]

        train_inds = ind0[ind0 <= last_inds[0]]
        val_inds = ind0[(ind0 > last_inds[0]) & (ind0 <= last_inds[1])]
        test_inds = ind0[ind0 > last_inds[1]]

        return train_inds, val_inds, test_inds
