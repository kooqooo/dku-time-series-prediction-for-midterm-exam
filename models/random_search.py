from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
from scipy.stats import randint

# RandomForest 모델 생성
rf = RandomForestRegressor()

# 하이퍼파라미터 랜덤 그리드 정의
param_dist = {
    'n_estimators': randint(10, 100),
    'max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],   # 트리의 최대 깊이
    'min_samples_split': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],       # 노드를 분할하기 위한 최소 샘플 수
    'min_samples_leaf': [1, 2, 4],         # 리프 노드의 최소 샘플 수
    'max_features': ['sqrt', 'log2']  # 각 노드 분할 시 고려할 최대 피처 수
}

# MAE를 사용한 스코어 정의 (MAE는 낮을수록 좋기 때문에 음수로 변환하여 최적화를 수행)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# RandomizedSearchCV 설정
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=100,                 # 탐색할 하이퍼파라미터 조합의 수
    scoring=mae_scorer,         # MAE 사용
    cv=5,                       # 교차 검증 5-폴드
    n_jobs=-1,                  # 병렬 처리
    verbose=2,
    random_state=42
)

# 학습 데이터로 랜덤 서치 실행
random_search.fit(X_train, y_train)

# 상위 5개의 최적 조합 출력
top_5_results = sorted(random_search.cv_results_['params'], key=lambda x: x['mean_test_score'], reverse=True)[:5]
top_5_scores = sorted(random_search.cv_results_['mean_test_score'], reverse=True)[:5]

print("Top 5 Best Parameter Combinations and Scores (MAE):")
for i, (params, score) in enumerate(zip(top_5_results, top_5_scores), 1):
    print(f"{i}. Parameters: {params}, Score (MAE): {-score}")
