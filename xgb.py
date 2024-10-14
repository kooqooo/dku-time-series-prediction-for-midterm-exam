from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
year = 2022

def convert_to_datetime(date_string):
    yymm = date_string[:4]
    time = date_string[5:]
    full_date_string = f"{year}{yymm}{time}"
    return datetime.strptime(full_date_string, "%Y%m%d%H:%M")

def drop_columns(columns, data):
    return data.drop(columns, axis=1)

# 데이터 불러오기
data = pd.read_csv('data/train.csv')
data["yymm"] = data["yymm"].apply(convert_to_datetime)
X = drop_columns(['Target', 'yymm'], data)
# X = drop_columns(['Target', 'yymm', 'V1', 'V2', 'V11', 'V3', 'V4', 'V5', 'V6', 'V8', 'V9', 'V7', 'V10', 'V12', 'V13'], data)
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# XGBoost 모델 생성 및 학습
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
score = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {score}")

# 특성 중요도 가져오기
importance = model.feature_importances_


# 특성 이름과 중요도를 결합
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importance})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# 시각화
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.xticks(rotation=90)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# 테스트 데이터 예측
test_data = pd.read_csv('data/test_set.csv')
test_data = drop_columns(['yymm'], test_data)
# test_data = drop_columns(['yymm', 'V1', 'V2', 'V11', 'V3', 'V4', 'V5', 'V6', 'V8', 'V9', 'V7', 'V10', 'V12', 'V13'], test_data)

model.fit(X, y)
predictions = model.predict(test_data)
test_data['predict'] = predictions
test_data['predict'].to_csv('result2.csv', index=False)
