# financial-data-generation

This is the GitHub repository for our 10-708 Final project on financial data generation using generative adversarial models.

Team:
* Olivier Filion
* Michael Agaby
* Owen Wang

## Data

Our data was downloaded from [Yahoo S&P 500](https://finance.yahoo.com/quote/%5EGSPC/history?period1=-315619200&period2=1640908800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true) and [Yahoo VIX](https://finance.yahoo.com/quote/%5EVIX/history?period1=-315619200&period2=1640908800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true). The downloaded csv files are in the [data folder](data). The data is then preprocessed using [features_engineering.py](feature_engineering.py) and the final dataset can be found in [data/features.csv](data/features.csv).

## Running the code

The Python notebook [midway_report.py](midway_report.py) contains the code to run our baseline model we used for our midway report. The notebook [gru_timegan.ipynb](gru_timegan.ipynb) contains the code to generate data using TimeGAN with GRU. This uses an implementation from [ydata-synthetic](https://github.com/ydataai/ydata-synthetic/blob/d888bcf3cb2c6e5b4f620f7e05816cfe54889d5d/src/ydata_synthetic/synthesizers/timeseries/timegan/model.py). The file [timegan.py] implements our transformer version of the TimeGAN model. Our PyTorch implementation of the training code follows closely the implementation from [ydata-synthetic](https://github.com/ydataai/ydata-synthetic/blob/d888bcf3cb2c6e5b4f620f7e05816cfe54889d5d/src/ydata_synthetic/synthesizers/timeseries/timegan/model.py).

The file [stats.py](stats.py) generates the visualizations and computes the summary statistics for all the datasets.
