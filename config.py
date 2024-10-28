random_state = 42
shuffle = True
# drop_columns = ["yymm"]
drop_columns = ["yymm"] + ['V1', 'V2', 'V3', 'V5', 'V6', 'V8', 'V9', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V18', 'V19', 'V20', 'V24', 'V26']

use_datetime = False
# scaler = "RobustScaler"
scaler = None

# RandomForest Hyperparameters
params = {
    "bootstrap": False,
    "criterion": "squared_error",
    "max_depth": 5,
    "max_features": "log2",
    "max_leaf_nodes": 8,
    "min_samples_leaf": 3,
    "min_samples_split": 15,
    "n_estimators": 45,
    "oob_score": False,
    "warm_start": True,
}

# params = {
#     "max_depth": 6,
#     "max_features": "log2",
#     "min_samples_leaf": 3,
#     "min_samples_split": 16,
#     "n_estimators": 45,
# }

# 시간 정보 추가
# params = {
#     "max_depth": 5,
#     "max_features": "log2",
#     "min_samples_leaf": 8,
#     "min_samples_split": 17,
#     "n_estimators": 80,
# }

# params = {
#     'bootstrap': False,
#     'ccp_alpha': 0.0,
#     'criterion': 'absolute_error',
#     'max_depth': 5,
#     'max_features': 'log2',
#     'max_leaf_nodes': 10,
#     'max_samples': None,
#     'min_impurity_decrease': 0.0,
#     'min_samples_leaf': 8,
#     'min_samples_split': 17,
#     'min_weight_fraction_leaf': 0.0,
#     'n_estimators': 80,
#     'oob_score': False,
#     'warm_start': True
# }

# params = {
#     "bootstrap": False,
#     "criterion": "squared_error",
#     "max_depth": 6,
#     "max_features": "log2",
#     "max_leaf_nodes": 14,
#     "min_samples_leaf": 8,
#     "min_samples_split": 17,
#     "n_estimators": 100,
#     "warm_start": True,
# }

# # 2024-10-26_21-47-18
# params = {
#     "bootstrap": False,
#     "criterion": "squared_error",
#     "max_depth": 7,
#     "max_features": "log2",
#     "max_leaf_nodes": 14,
#     "min_samples_leaf": 12,
#     "min_samples_split": 15,
#     "n_estimators": 100,
#     "warm_start": True,
# }

# # 2024-10-26_21-48-49
# params = {
#     "bootstrap": False,
#     "criterion": "absolute_error",
#     "max_depth": 7,
#     "max_features": "log2",
#     "max_leaf_nodes": 14,
#     "min_samples_leaf": 12,
#     "min_samples_split": 15,
#     "n_estimators": 100,
#     "warm_start": True,
# }

# # 1차
# params = {
#     "bootstrap": False,
#     "criterion": "squared_error",
#     "max_depth": 7,
#     "max_features": "log2",
#     "max_leaf_nodes": 14,
#     "min_samples_leaf": 5,
#     "min_samples_split": 17,
#     "n_estimators": 45,
#     "oob_score": False,
#     "warm_start": True,
# }


# # 2차
# params = {
#     "bootstrap": False,
#     "criterion": "absolute_error",
#     "max_depth": 7,
#     "max_features": "log2",
#     "max_leaf_nodes": 10,
#     "min_samples_leaf": 8,
#     "min_samples_split": 15,
#     "n_estimators": 80,
#     "oob_score": False,
#     "warm_start": True,
# }

# params = {
#     "bootstrap": False,
#     "criterion": "absolute_error",
#     "max_depth": 6,
#     "max_features": "log2",
#     "max_leaf_nodes": 10,
#     "min_samples_leaf": 8,
#     "min_samples_split": 17,
#     "n_estimators": 45,
#     "oob_score": False,
#     "warm_start": True,
# }

