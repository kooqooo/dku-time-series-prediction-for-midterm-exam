import os

import pandas as pd

from utils.preprocess import convert_to_datetime

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def load_train_data():
    data_path = os.path.join(data_dir, "train.csv")
    df = pd.read_csv(data_path)
    df["yymm"] = df["yymm"].map(convert_to_datetime)
    X = df.drop('Target', axis=1)
    y = df['Target']

    return X, y

def load_test_data():
    data_path = os.path.join(data_dir, "test_set.csv")
    df = pd.read_csv(data_path)
    df["yymm"] = df["yymm"].map(convert_to_datetime)

    return X

if __name__ == "__main__":
    X, y = load_train_data()
    print(X.head())
    print(y.head())

