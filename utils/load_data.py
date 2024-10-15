import os

import pandas as pd
from sklearn.model_selection import train_test_split

from config import *
from utils.preprocess import convert_to_datetime

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_train_data():  # train.py에서 사용
    data_path = os.path.join(data_dir, "train.csv")
    df = pd.read_csv(data_path)
    df["yymm"] = df["yymm"].map(convert_to_datetime)
    X = df.drop("Target", axis=1)
    y = df["Target"]

    return X, y


def load_test_data():  # inference.py에서 사용
    data_path = os.path.join(data_dir, "test_set.csv")
    df = pd.read_csv(data_path)
    df["yymm"] = df["yymm"].map(convert_to_datetime)

    return df


def split_data(X, y):
    return train_test_split(
        X, y, test_size=0.2, random_state=random_state, shuffle=shuffle
    )


if __name__ == "__main__":
    X, y = load_train_data()
    print(X.head())
    print(y.head())
