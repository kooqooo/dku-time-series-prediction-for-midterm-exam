import pandas as pd
import matplotlib.pyplot as plt

from utils.load_data import load_train_data
from utils.preprocess import drop_columns

X, y = load_train_data()
