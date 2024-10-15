import config
from utils.load_data import load_train_data
from utils.preprocess import drop_columns
from utils.time_utils import time_wrapper

X, y = load_train_data()
X = drop_columns(X, columns=config.columns)

@time_wrapper
def train(X, y, model_name):
    print(f"======== {model_name} ========")
    if model_name in ["XGBoost", "XGB", "xgb", "xgboost"]:
        from xgboost import XGBRegressor
        model = XGBRegressor(random_state=config.random_state)  # 모델 정의 부분을 분리할 예정
    elif model_name in ["RandomForest", "RF", "rf", "randomforest"]:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(random_state=config.random_state, n_jobs=-1, **config.params)
    else:
        raise ValueError("Invalid model_name")
    
    model.fit(X, y)
    return model
