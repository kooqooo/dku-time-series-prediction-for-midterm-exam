from datetime import datetime

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