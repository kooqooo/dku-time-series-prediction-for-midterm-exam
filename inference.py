import config
from train import X, train, y
from utils.load_data import load_test_data
from utils.preprocess import drop_columns, get_scaler
from utils.time_utils import get_current_time

X_train = X
test_data = load_test_data()
test_data = drop_columns(test_data, columns=config.columns)
print(test_data.head())

if config.scaler:
    print(f"사용된 scaler: {config.scaler}")
    scaler = get_scaler(config.scaler)

    X_train[:] = scaler.fit_transform(X[:])
    test_data[:] = scaler.transform(test_data[:])

model = train(X_train, y, model_name="RandomForest")
predictions = model.predict(test_data)
test_data["predict"] = predictions
test_data["predict"].to_csv(f"{get_current_time()}.csv", index=False)
