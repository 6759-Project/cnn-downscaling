# cnn-downscaling
Downscaling global climate models with convolutional neural networks

## Setup
 * Download processed* historical data from [Google Drive](https://drive.google.com/drive/folders/1_ML9zY6t6W4jSg7jzN8SpERxfkVZBSBJ?usp=sharing) (click "Download All")
 * Move the downloaded data to a new "./data/" subfolder in this repository in order to use scripts with default configurations. The final path should be "/path/to/repo/data/processed/..."
 * Install the corresponding conda environment via environment.yml via `conda env create --name {your env name here} --file=environment.yml`

\* Raw data can be downloaded by following the instructions on the [WeatherBench repo](https://github.com/pangeo-data/WeatherBench#download-the-data). We downloaded `2m_temperature` and `total_precipitation` at 1.4 and 5.625 degrees of resolution


## Training
All experiments ran for this project can be replicated with src/training/experiments.sh
