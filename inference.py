import config
from train import X, train, y
from utils.load_data import load_test_data
from utils.preprocess import drop_columns
from utils.time_utils import get_current_time

test_data = load_test_data()
test_data = drop_columns(test_data, columns=config.columns)
print(test_data.head())

model = train(X, y, model_name="RandomForest")
predictions = model.predict(test_data)
test_data["predict"] = predictions
test_data["predict"].to_csv(f"{get_current_time()}.csv", index=False)
