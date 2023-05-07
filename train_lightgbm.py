import lightgbm as lgb

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def _train_model(model_state_output_path: str = "tmp/model.txt", seed=999):
    """Function to train an LightGBM Classifier using the sklearn Iris dataset"""
    dataset = load_iris()
    x_train, _, y_train, _ = train_test_split(
        dataset["data"], dataset["target"], test_size=0.2, random_state=seed
    )
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "verbosity": -1
    }
    train_data = lgb.Dataset(x_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=3)

    model.save_model(model_state_output_path)
    return model


_train_model()