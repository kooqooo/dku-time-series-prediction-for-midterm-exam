from time import time

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split

# 데이터 불러오기
df = pd.read_csv("data/train.csv")

# 'yymm' 컬럼 제거
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
    'max_depth': [*range(1, 10)],   # 트리의 최대 깊이
    'max_features': ['log2', 'sqrt'],   # 각 트리를 분할할 때 고려할 최대 feature 수
    'min_samples_leaf': [*range(1, 10)],    # 리프 노드에 있어야 하는 최소 샘플 수
    'min_samples_split': [*range(10, 21), 30],  # 내부 노드를 분할하기 위한 최소 샘플 수
    'n_estimators': [*range(45, 66, 5), 80],    # 결정 트리의 개수
}

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
print(f"Time: {time() - start:.2f}")