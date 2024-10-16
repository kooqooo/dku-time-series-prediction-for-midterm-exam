random_state = 42
shuffle = False
columns = ["yymm"]
scaler = "RobustScaler"
scaler = None

## RandomForest Hyperparameters
params = {
    "max_depth": 6,
    "max_features": "log2",
    "min_samples_leaf": 3,
    "min_samples_split": 16,
    "n_estimators": 45,
}
