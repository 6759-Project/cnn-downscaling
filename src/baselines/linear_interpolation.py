import numpy as np
import xarray as xr
from tqdm import tqdm

from scipy.interpolate import interp2d
from data import Data



def main():
    # Loading data
    data = Data(
        coarse_dir="data/processed/temp/5625/temp_5625_processed.zarr",
        fine_dir="data/processed/temp/1406/temp_1406_processed.zarr"
        )

    data.split()

    x_coarse = data.coarse_test.lat.values.copy()
    y_coarse = data.coarse_test.lon.values.copy()

    x_fine = data.fine_test.lat.values.copy()
    y_fine = data.fine_test.lon.values.copy()

    predictions = []
    for i in tqdm(range(data.n_test)):

        z_coarse = data.coarse_test.isel({"date":i}).t2m.values.copy()

        inter_fun = interp2d(x_coarse, y_coarse, z_coarse, kind='linear')

        predictions.append(inter_fun(x_fine, y_fine))

    preds = np.stack(predictions, axis=0)

    # find correlation
    trues = data.fine_test.t2m.values.copy()

    corrs=[]
    for i in tqdm(range(len(preds))):
        corrs.append(np.corrcoef(preds[i].flatten(), trues[i].flatten())[0,1])
    print(f'corrs: {np.mean(corrs)} +/- {np.std(corrs)}' )

    # mean square root error
    mse=[]
    for i in tqdm(range(len(preds))):
        mse.append(
            np.mean((preds[i] - trues[i])**2)
        )
    print(f'mse: {np.mean(mse)} +/- {np.std(mse)}' )

    # Saving the array as zarr
    zarr_preds = xr.Dataset(data_vars={'t2m':(["date", "lat", "lon"], preds)}, coords=data.fine_test.coords)
    zarr_preds.to_zarr('test_hr_pred_bilinearInterpolation.zarr')
    data.fine_test.to_zarr('test_hr_true_bilinearInterpolation.zarr')

if __name__ == '__main__':
    main()