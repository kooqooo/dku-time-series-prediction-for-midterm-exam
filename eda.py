from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# 데이터 로드 (여기서는 이전에 처리한 df를 사용한다고 가정합니다)
df = pd.read_csv("data/train.csv")  # 필요하다면 이 줄을 사용하여 데이터를 로드하세요

# # 기본 정보 확인
# print(df.info())
# print("\nNull 값 확인:")
# print(df.isnull().sum())

# print("\n기술 통계:")
# print(df.describe())

year = 2022


def convert_to_datetime(date_string):
    yymm = date_string[:4]
    time = date_string[5:]
    full_date_string = f"{year}{yymm}{time}"
    return datetime.strptime(full_date_string, "%Y%m%d%H:%M")


df["datetime"] = df["yymm"].apply(convert_to_datetime)


# # 시계열 플롯
# plt.figure(figsize=(25, 5))
# plt.plot(df['datetime64'], df['Target'])  # 'Target'을 실제 타겟 변수 이름으로 변경하세요
# plt.title('Time Series Plot')
# plt.xlabel('Date')
# plt.ylabel('Target Variable')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # 주요 특성들의 분포 시각화
# numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
# n_features = len(numerical_features)
# n_rows = (n_features + 1) // 2

# plt.figure(figsize=(15, 5*n_rows))
# for i, feature in enumerate(numerical_features, 1):
#     plt.subplot(n_rows, 2, i)
#     sns.histplot(df[feature], kde=True)
#     plt.title(f'Distribution of {feature}')
# plt.tight_layout()
# plt.show()

# # 상관관계 히트맵
# correlation_matrix = df[numerical_features].corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title('Correlation Heatmap')
# plt.tight_layout()
# plt.show()

# # 요일별 타겟 변수 평균
# df['dayofweek'] = df['datetime64'].dt.dayofweek
# df['dayofweek_name'] = df['datetime64'].dt.day_name()
# daily_avg = df.groupby('dayofweek_name')['Target'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
# plt.figure(figsize=(10, 6))
# daily_avg.plot(kind='bar')
# plt.title('Average Target Variable by Day of Week')
# plt.ylabel('Average Target')
# plt.tight_layout()
# plt.show()

# # 시간대별 타겟 변수 평균
# hourly_avg = df.groupby('hour')['Target'].mean()
# plt.figure(figsize=(12, 6))
# hourly_avg.plot(kind='line', marker='o')
# plt.title('Average Target Variable by Hour')
# plt.xlabel('Hour')
# plt.ylabel('Average Target')
# plt.tight_layout()
# plt.show()

# # 계절성 분해 (일일 데이터가 충분한 경우)
# if len(df) >= 2 * 24 * 7:  # 최소 2주 데이터
#     df_daily = df.set_index('datetime64').resample('D')['Target'].mean()
#     result = seasonal_decompose(df_daily, model='additive', period=7)
#     fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 20))
#     result.observed.plot(ax=ax1)
#     ax1.set_title('Observed')
#     result.trend.plot(ax=ax2)
#     ax2.set_title('Trend')
#     result.seasonal.plot(ax=ax3)
#     ax3.set_title('Seasonal')
#     result.resid.plot(ax=ax4)
#     ax4.set_title('Residual')
#     plt.tight_layout()
#     plt.show()

# # 주요 특성과 타겟 변수의 산점도
# for feature in numerical_features[:5]:  # 처음 5개 특성에 대해서만
#     plt.figure(figsize=(10, 6))
#     plt.scatter(df[feature], df['Target'], alpha=0.5)
#     plt.title(f'Scatter plot of {feature} vs Target')
#     plt.xlabel(feature)
#     plt.ylabel('Target Variable')
#     plt.tight_layout()
#     plt.show()


# # 10분 단위의 시간 특성 추출
# df['time_of_day'] = df['datetime'].dt.hour * 6 + df['datetime'].dt.minute // 10

# # 하루를 144개의 10분 간격으로 나눕니다 (24시간 * 6)
# df['time_of_day_sin'] = np.sin(df['time_of_day'] * (2 * np.pi / 144))
# df['time_of_day_cos'] = np.cos(df['time_of_day'] * (2 * np.pi / 144))

# # 요일 특성 추출
# df['day_of_week'] = df['datetime'].dt.dayofweek

# # 타겟 변수 (예: 'value')의 평균을 계산
# daily_pattern = df.groupby('time_of_day')['Target'].mean().reset_index()

# # 시각화
# plt.figure(figsize=(12, 6))
# plt.plot(daily_pattern['time_of_day'], daily_pattern['Target'])
# plt.title('Average Daily Pattern')
# plt.xlabel('Time of Day (10-minute intervals)')
# plt.ylabel('Average Value')
# plt.xticks(np.arange(0, 145, 12), [f'{h:02d}:00' for h in range(0, 25, 2)])
# plt.grid(True)
# plt.show()

# # 요일별 패턴 시각화
# plt.figure(figsize=(14, 8))
# for day in range(7):
#     day_data = df[df['day_of_week'] == day]
#     day_pattern = day_data.groupby('time_of_day')['Target'].mean()
#     plt.plot(day_pattern.index, day_pattern.values, label=f'Day {day}')

# plt.title('Daily Patterns by Day of Week')
# plt.xlabel('Time of Day (10-minute intervals)')
# plt.ylabel('Average Value')
# plt.xticks(np.arange(0, 145, 12), [f'{h:02d}:00' for h in range(0, 25, 2)])
# plt.legend()
# plt.grid(True)
# plt.show()

# # 시계열 분해
# # 일별 데이터로 리샘플링
# daily_data = df.set_index('datetime').resample('D')['Target'].mean()

# # 주기성 분해
# result = seasonal_decompose(daily_data, model='additive', period=7)

# # 결과 시각화
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
# result.observed.plot(ax=ax1)
# ax1.set_title('Observed')
# result.trend.plot(ax=ax2)
# ax2.set_title('Trend')
# result.seasonal.plot(ax=ax3)
# ax3.set_title('Seasonal')
# result.resid.plot(ax=ax4)
# ax4.set_title('Residual')
# plt.tight_layout()
# plt.show()

# # 자기상관 분석
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# print(df.info())
# df = df.drop("yymm", axis=1)
# df = df.drop("datetime", axis=1)
# plt.figure(figsize=(12, 6))
# plot_acf(df["Target"], lags=144, ax=plt.gca())  # 24시간 * 6 (10분 간격)
# plt.title("Autocorrelation Function")
# plt.show()

# # 부분 자기상관 분석
# plt.figure(figsize=(12, 6))
# plot_pacf(df["Target"], lags=144, ax=plt.gca())
# plt.title("Partial Autocorrelation Function")
# plt.show()

from utils.load_data import load_test_data
test = load_test_data()
columns = df.columns.to_list()[1:-2]
for column in columns:
    print(column)
    print(f"(train) {column} 최대: {df[column].max()}, 최소: {df[column].min()}")
    print(f"(test) {column} 최대: {test[column].max()}, 최소: {test[column].min()}")
    print()
