import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core import Dataset, Workspace
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.datasets import make_regression
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

def clean_data(data):
    # Dict for converting categorical data to numerical data
    cut = {"Fair":0, "Good":1, "Very Good":2, "Ideal":3, "Premium":4}
    colour = {"D":0, "E":1, "F":2, "G":3, "H":4, "I":5, "J":6}
    clarity = {"SI2":0, "SI1":1, "VS1":1, "VS2":2, "VVS2":3, "VVS1":4, "I1":5, "IF":6}

    # Clean and encode the data
    x_df = data.to_pandas_dataframe().dropna()
    x_df.describe()

    # Remove the first column as this just contains a index for each sample
    x_df.pop("Column2")

    x_df["cut"] = x_df.cut.map(cut)
    x_df["color"] = x_df.color.map(colour)
    x_df["clarity"] = x_df.clarity.map(clarity)

    y_df = x_df.pop("price")
    return x_df, y_df


def main():

    run = Run.get_context()

    # get input dataset by name
    ws = run.experiment.workspace

    key = 'diamond-data'

    if key in ws.datasets.keys(): 
        found = True
        ds = ws.datasets[key] 
    
    x, y = clean_data(ds)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 20)
    
    # Load the model
    from joblib import Parallel, delayed
    import joblib
    hgbm = joblib.load('./hyperdriveModel/HGBM.pkl')

    # Test the model
    y_predict = hgbm.predict(x_test)
    print(y_predict[:10])

    from sklearn.metrics import mean_squared_error
    from math import sqrt

    y_actual = y_test.values.flatten().tolist()
    rmse = sqrt(mean_squared_error(y_actual, y_predict))
    rmse

    sum_actuals = sum_errors = 0

    for actual_val, predict_val in zip(y_actual, y_predict):
        abs_error = actual_val - predict_val
        if abs_error < 0:
            abs_error = abs_error * -1

        sum_errors = sum_errors + abs_error
        sum_actuals = sum_actuals + actual_val

    mean_abs_percent_error = sum_errors / sum_actuals
    print("Model MAPE:")
    print(mean_abs_percent_error)
    print()
    print("Model Accuracy:")
    print(1 - mean_abs_percent_error)


if __name__ == '__main__':
    main()
