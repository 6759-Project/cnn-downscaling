import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from data import Data

def main():
    # Loading data
    data = Data(
        coarse_dir="data/processed/temp/5625/temp_5625_processed.zarr",
        fine_dir="data/processed/temp/1406/temp_1406_processed.zarr"
        )

    data.split()

    # import ipdb; ipdb.set_trace()

    coarse = data.coarse_test.isel({'date':0}).t2m.to_numpy()
    fine = data.fine_test.isel({'date':0}).t2m.to_numpy()

    plt.imshow(coarse)
    plt.figure()
    plt.imshow(fine)
    plt.show()
if __name__ == '__main__':
    main()