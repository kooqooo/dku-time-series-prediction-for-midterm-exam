from time import time
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split


year = 2021

def convert_to_datetime(date_string: str):
    yymm = date_string[:4]
    time = date_string[5:]
    full_date_string = f"{year}{yymm}{time}"
    return datetime.strptime(full_date_string, "%Y%m%d%H:%M")

def add_time_features(df: pd.DataFrame, datetime_column: str = 'yymm'):
    df['month'] = df[datetime_column].dt.month
    df['day'] = df[datetime_column].dt.day
    df['hour'] = df[datetime_column].dt.hour
    df['weekday'] = df[datetime_column].dt.weekday
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    return df

# 데이터 불러오기
df = pd.read_csv("data/train.csv")

# 'yymm' 컬럼 제거
df["yymm"] = df["yymm"].map(convert_to_datetime)
df = add_time_features(df)
df = df.drop(columns=["yymm"])

# 특성과 타겟 분리
X = df.drop(columns=["Target"])
y = df["Target"]

# 훈련 및 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 랜덤 포레스트 회귀 모델 설정
rf = RandomForestRegressor(random_state=42)

param_grid = {
    # 'max_depth': [*range(3, 9)],   # 트리의 최대 깊이
    # 'max_features': ['log2'],   # 각 트리를 분할할 때 고려할 최대 feature 수
    # 'min_samples_leaf': [*range(1, 10)],    # 리프 노드에 있어야 하는 최소 샘플 수
    # 'min_samples_split': [*range(10, 21)],  # 내부 노드를 분할하기 위한 최소 샘플 수
    # 'n_estimators': [*range(45, 100, 5)],    # 결정 트리의 개수
    "max_depth": [5, 6, 7, 8],
    "max_features": ["log2"],
    "min_samples_leaf": [3, 5, 8, 12],
    "min_samples_split": [15, 16, 17, 18],
    "n_estimators": [45, 50, 80, 100],
    'criterion': ['squared_error', 'absolute_error'],  # 분할 품질을 측정하는 기준
    # 'criterion': ['squared_error'],  # 분할 품질을 측정하는 기준
    'bootstrap': [False],  # 부트스트랩 샘플링 사용 여부
    'warm_start': [True, False],  # 이전 호출의 솔루션을 재사용하여 학습을 추가할지 여부
    'max_leaf_nodes': [8, 10, 12, 14],  # 리프 노드의 최대 개수
    'oob_score': [True, False],  # out-of-bag 샘플을 사용하여 일반화 오차 추정
}

# param_grid = {
#     'max_depth': [*range(3, 6)],   # 트리의 최대 깊이
# }

start = time()
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=5,
    n_jobs=-1,
    verbose=2,
)
grid_search.fit(X_train, y_train)

# 최적 하이퍼파라미터 출력
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# 최적 모델로 예측
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)


# MAE 평가
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)
total_time = time() - start
print(f"소요 시간: {total_time//60}분 {total_time%60}초")

print("="*80)
results = sorted(
    zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']),
    key=lambda x: x[1], 
    reverse=True
)[:5]
from pprint import pprint
for params, score in results:
    print(f"Parameters: {params}, Score: {-score}")
    pprint(params)

