from config import *
from utils.load_data import load_train_data
from utils.preprocess import drop_columns
from utils.time_wrapper import time_wrapper

X, y = load_train_data()
X = drop_columns(X, columns=columns)

@time_wrapper
def train(X, y, model_name):
    print(f"======== {model_name} ========")
    if model_name in ["XGBoost", "XGB", "xgb", "xgboost"]:
        from xgboost import XGBRegressor
        model = XGBRegressor(random_state=random_state)
    else:
        raise ValueError("Invalid model_name")
    
    model.fit(X, y)
    return model
