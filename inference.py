import config
from train import X, train, y
from utils.load_data import load_test_data
from utils.preprocess import drop_columns, get_scaler, add_time_features
from utils.time_utils import get_current_time

X_train = X
test_data = load_test_data()
if config.use_datetime:
    test_data = add_time_features(test_data)
test_data = drop_columns(test_data, columns=config.drop_columns)

print(X_train.head())
print(test_data.head())

if config.scaler:
    print(f"사용된 scaler: {config.scaler}")
    scaler = get_scaler(config.scaler)

    X_train[:] = scaler.fit_transform(X[:])
    test_data[:] = scaler.transform(test_data[:])

model = train(X_train, y, model_name="RandomForest") # RandomForest 모델 사용
predictions = model.predict(test_data)
test_data["predict"] = predictions
test_data["predict"].to_csv(f"output/{get_current_time()}_{config.random_state}.csv", index=False)
