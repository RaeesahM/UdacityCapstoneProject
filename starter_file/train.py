
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core import Dataset
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

    # Add arguments to the script
    parser = argparse.ArgumentParser()

    parser.add_argument("--LR", type = float, default = 0.01, help = "Learning rate for the HGBM")
    parser.add_argument("--maxDepth", type = int, default = 10, help = "Maximum tree depth")

    # parser.add_argument("--input-data", type=str)

    args = parser.parse_args()

    run = Run.get_context()

    run.log("LR:", np.float(args.LR))
    run.log("maxDepth:", np.int(args.maxDepth))

    # get input dataset by name
    ws = run.experiment.workspace

    key = 'diamond-data'

    if key in ws.datasets.keys(): 
        found = True
        ds = ws.datasets[key] 


    print(ds)
    
    x, y = clean_data(ds)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 20)
    # print(x_train)
    # print(x_train.shape)
    categorical_mask =  [False, True, True, True, False, False,False, False, False]

    import sklearn
    print(sklearn.__version__)

    gbm = HistGradientBoostingRegressor(learning_rate = args.LR,
                                        max_depth = args.maxDepth, 
                                        categorical_features=categorical_mask)
    gbm.fit(x_train, y_train)

    mse = gbm.score(x_test, y_test)
    run.log("Mean Squared Error", np.float(mse))

    # Save the model so that it can be registered later
    from joblib import Parallel, delayed
    import joblib
  
  
    # Save the model as a pickle in a file
    path = './outputs/log'

    if not os.path.isdir(path):
        os.makedirs(path)
    joblib.dump(gbm, './outputs/log/HGBM.pkl')


if __name__ == '__main__':
    main()
