import os
import numpy as np
import xarray as xr
from typing import List

class Data:
    """ Loads zarr files and returns them.

        Assumes highly-processed data can fit in memory.
    """

    def __init__(self, coarse_dir: os.PathLike, fine_dir: os.PathLike):
        """
        Arguments:
          * coarse_dir: the path to the coarse-grid data
          * fine_dir: the path to the fine-grid data
        """

        self.coarse = xr.open_zarr(coarse_dir)
        self.fine = xr.open_zarr(fine_dir)

        self.coarse_train = None
        self.fine_train = None

        self.coarse_valid = None
        self.fine_valid = None

        self.coarse_test = None
        self.fine_test = None

        self.n_total = None
        self.n_train = None
        self.n_valid = None
        self.n_test = None

        self.daily_means = None
        self.daily_std = None

        if not (self.coarse.date == self.fine.date).all():
            raise ValueError("coarse-grid dates do not match fine-grid dates")

    def split(self, ratios:List[float]=[0.7, 0.2], n_total=None):

        if n_total is None:
            self.n_total = len(self.fine.date)
        else:
            self.n_total = n_total

        self.n_train = int(ratios[0]*self.n_total)
        self.n_valid = int(ratios[1]*self.n_total)
        self.n_test = self.n_total-self.n_train-self.n_valid

        self.coarse_train = self.coarse.isel(date=slice(0, self.n_train))
        self.fine_train = self.fine.isel(date=slice(0, self.n_train))

        self.coarse_valid = self.coarse.isel(date=slice(self.n_train, self.n_train+self.n_valid))
        self.fine_valid = self.fine.isel(date=slice(self.n_train, self.n_train+self.n_valid))

        self.coarse_test = self.coarse.isel(date=slice(self.n_train+self.n_valid, self.n_train+self.n_valid+self.n_test))
        self.fine_test = self.fine.isel(date=slice(self.n_train+self.n_valid, self.n_train+self.n_valid+self.n_test))

    def standardize(self):
        self.daily_means = self.coarse.mean(["lat", "lon"])
        self.daily_std = self.coarse.std(["lat", "lon"])
        self.coarse = (self.coarse - self.daily_means) / self.daily_std
        self.fine = (self.fine - self.daily_means) / self.daily_std

    def destandardize(self, data: xr.Dataset) -> xr.Dataset:
        return (data * self.daily_std) + self.daily_means