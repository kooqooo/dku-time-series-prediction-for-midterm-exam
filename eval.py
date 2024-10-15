from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import config
from train import X, train, y

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=config.random_state, shuffle=config.shuffle
)


def evaluate(model_name):
    model = train(X_train, y_train, model_name=model_name)
    y_pred = model.predict(X_valid)

    print(f"사용한 columns의 수 : {len(X.columns)}")
    print(f"MAE: {mean_absolute_error(y_valid, y_pred):.5f}")
    print()


if __name__ == "__main__":
    evaluate("XGBoost")
    evaluate("RandomForest")
