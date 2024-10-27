import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

import config
from utils.load_data import load_train_data
from utils.preprocess import drop_columns
from utils.time_utils import time_wrapper

X, y = load_train_data()
X = drop_columns(X, columns=config.columns)

# 훈련/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의
model = RandomForestRegressor(random_state=config.random_state, n_jobs=-1, **config.params)

# ### 1. Optimal Feature Selection using Sequential Feature Selector
# print("== Finding Optimal Features ==")

# feature_scores = []
# for k in tqdm(range(1, X_train.shape[1])):
#     sfs = SequentialFeatureSelector(model, n_features_to_select=k, direction='forward', cv=5, scoring='neg_mean_absolute_error')
#     sfs.fit(X_train, y_train)
#     score = cross_val_score(model, sfs.transform(X_train), y_train, cv=5, scoring='neg_mean_absolute_error').mean()
#     feature_scores.append((k, score))

# # 최적의 피처 개수 선택
# best_k = max(feature_scores, key=lambda x: x[1])[0]
# print(f"Optimal number of features: {best_k}")

# # 최적의 피처로 데이터 변환
# sfs_optimal = SequentialFeatureSelector(model, n_features_to_select=best_k, direction='forward', cv=5, scoring='neg_mean_absolute_error')
# sfs_optimal.fit(X_train, y_train)

# selected_features = X_train.columns[sfs_optimal.get_support()]
# print(f"Selected Optimal Features: {selected_features.tolist()}")

# # 최적의 피처로 훈련 및 테스트 데이터셋 변환
# X_train_optimal = sfs_optimal.transform(X_train)
# X_test_optimal = sfs_optimal.transform(X_test)

# ### 2. RandomForest with Optimal Features
# print("\n== GPT RandomForest with Optimal Features ==")

# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train_optimal, y_train)

# # K-Fold 교차 검증 (cv=5)
# rf_cv_scores = cross_val_score(rf_model, X_train_optimal, y_train, cv=5, scoring='neg_mean_absolute_error')
# mean_mae_rf = -np.mean(rf_cv_scores)
# print(f"Cross-Validated MAE (Random Forest, Optimal Features): {mean_mae_rf}")

# # 최적의 피처로 테스트 데이터 예측 및 최종 MAE
# y_pred_rf = rf_model.predict(X_test_optimal)
# mae_rf = mean_absolute_error(y_test, y_pred_rf)
# print(f"Test MAE (GPT Random Forest, Optimal Features): {mae_rf}")

# print("\n== GPT RandomForest with All Features ==")
# rf_model.fit(X_train, y_train)

# # K-Fold 교차 검증 (cv=5)
# rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
# mean_mae_rf = -np.mean(rf_cv_scores)
# print(f"Cross-Validated MAE (GPT Random Forest, All Features): {mean_mae_rf}")

# print("\n== My Random Forest with Optimal Features ==")
# # 랜덤 포레스트 모델 훈련
# rf_model = RandomForestRegressor(random_state=config.random_state, n_jobs=-1, **config.params)
# rf_model.fit(X_train_optimal, y_train)

# # K-Fold 교차 검증 (cv=5)
# rf_cv_scores = cross_val_score(rf_model, X_train_optimal, y_train, cv=5, scoring='neg_mean_absolute_error')
# mean_mae_rf = -np.mean(rf_cv_scores)
# print(f"Cross-Validated MAE (My Random Forest, Optimal Features): {mean_mae_rf}")

# print("\n== My Random Forest with All Features ==")
# rf_model.fit(X_train, y_train)

# # K-Fold 교차 검증 (cv=5)
# rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
# mean_mae_rf = -np.mean(rf_cv_scores)
# print(f"Cross-Validated MAE (My Random Forest, All Features): {mean_mae_rf}")


# # from matplotlib import pyplot as plt

# 랜덤 포레스트 모델 훈련
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model = RandomForestRegressor(random_state=config.random_state, n_jobs=-1, **config.params)
rf_model.fit(X_train, y_train)

# 피처 중요도 시각화
from matplotlib import pyplot as plt
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.show()

# # 성능 측정 결과 그래프
# ks, scores = zip(*feature_scores)
# plt.plot(ks, [-s for s in scores], marker='o')
# plt.xlabel("Number of Features")
# plt.ylabel("Mean Absolute Error (MAE)")
# plt.title("Elbow Method for Optimal Feature Selection")
# plt.show()

# 사용할 피처 개수
k = 4

### 1. 필터 방식 (Filter Method)
print("== Filter Method ==")
# 상관계수 기반으로 피처 선택
filter_selector = SelectKBest(score_func=f_regression, k=k)
X_train_filter = filter_selector.fit_transform(X_train, y_train)
X_test_filter = filter_selector.transform(X_test)

# 선택된 피처와 제거된 피처 출력
selected_features_filter = X_train.columns[filter_selector.get_support()]
removed_features_filter = X_train.columns[~filter_selector.get_support()]
print(f"Selected Features (Filter): {selected_features_filter.tolist()}")
print(f"Removed Features (Filter): {removed_features_filter.tolist()}")

# MAE 계산 (cv=5)
mae_filter = -cross_val_score(model, X_train_filter, y_train, cv=5, scoring='neg_mean_absolute_error').mean()
print(f"MAE (Filter Method): {mae_filter}")

### 2. 순방향 선택 (Forward Selection)
print("\n== Forward Selection ==")
# 순방향 선택
sfs_forward = SequentialFeatureSelector(model, n_features_to_select=k, direction='forward', scoring='neg_mean_absolute_error', cv=5)
sfs_forward.fit(X_train, y_train)

X_train_forward = sfs_forward.transform(X_train)
X_test_forward = sfs_forward.transform(X_test)

# 선택된 피처와 제거된 피처 출력
selected_features_forward = X_train.columns[sfs_forward.get_support()]
removed_features_forward = X_train.columns[~sfs_forward.get_support()]
print(f"Selected Features (Forward): {selected_features_forward.tolist()}")
print(f"Removed Features (Forward): {removed_features_forward.tolist()}")

# MAE 계산 (cv=5)
mae_forward = -cross_val_score(model, X_train_forward, y_train, cv=5, scoring='neg_mean_absolute_error').mean()
print(f"MAE (Forward Selection): {mae_forward}")

### 3. 후방 제거 (Backward Elimination)
print("\n== Backward Elimination ==")
# 후방 제거
sfs_backward = SequentialFeatureSelector(model, n_features_to_select=k, direction='backward', scoring='neg_mean_absolute_error', cv=5)
sfs_backward.fit(X_train, y_train)

X_train_backward = sfs_backward.transform(X_train)
X_test_backward = sfs_backward.transform(X_test)

# 선택된 피처와 제거된 피처 출력
selected_features_backward = X_train.columns[sfs_backward.get_support()]
removed_features_backward = X_train.columns[~sfs_backward.get_support()]
print(f"Selected Features (Backward): {selected_features_backward.tolist()}")
print(f"Removed Features (Backward): {removed_features_backward.tolist()}")

# MAE 계산 (cv=5)
mae_backward = -cross_val_score(model, X_train_backward, y_train, cv=5, scoring='neg_mean_absolute_error').mean()
print(f"MAE (Backward Elimination): {mae_backward}")

### 4. Recursive Feature Elimination (RFE)
print("\n== Recursive Feature Elimination (RFE) ==")
# RFE
rfe_selector = RFE(estimator=model, n_features_to_select=k, step=1)
rfe_selector.fit(X_train, y_train)

X_train_rfe = rfe_selector.transform(X_train)
X_test_rfe = rfe_selector.transform(X_test)

# 선택된 피처와 제거된 피처 출력
selected_features_rfe = X_train.columns[rfe_selector.get_support()]
removed_features_rfe = X_train.columns[~rfe_selector.get_support()]
print(f"Selected Features (RFE): {selected_features_rfe.tolist()}")
print(f"Removed Features (RFE): {removed_features_rfe.tolist()}")

# MAE 계산 (cv=5)
mae_rfe = -cross_val_score(model, X_train_rfe, y_train, cv=5, scoring='neg_mean_absolute_error').mean()
print(f"MAE (RFE): {mae_rfe}")

# 결과 비교
print("\n=== MAE Comparison ===")
print(f"Filter Method MAE: {mae_filter}")
print(f"Forward Selection MAE: {mae_forward}")
print(f"Backward Elimination MAE: {mae_backward}")
print(f"RFE MAE: {mae_rfe}")