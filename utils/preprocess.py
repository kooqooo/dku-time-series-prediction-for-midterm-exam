from datetime import datetime

import pandas as pd
import numpy as np

year = 2021


def convert_to_datetime(date_string: str):
    yymm = date_string[:4]
    time = date_string[5:]
    full_date_string = f"{year}{yymm}{time}"
    return datetime.strptime(full_date_string, "%Y%m%d%H:%M")


def drop_columns(data, columns: list[str]):
    if not columns:
        return data
    return data.drop(columns, axis=1)

def get_scaler(scaler_name: str):
    if not scaler_name:
        return None
    if scaler_name == "StandardScaler":
        from sklearn.preprocessing import StandardScaler
        return StandardScaler()
    elif scaler_name == "RobustScaler":
        from sklearn.preprocessing import RobustScaler
        return RobustScaler()
    else:
        raise ValueError("Invalid scaler name")
    
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