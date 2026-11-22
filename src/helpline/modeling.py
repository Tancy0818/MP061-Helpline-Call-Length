"""The complete preprocessing/model pipeline used for training and serving."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(candidate, settings):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    transform = ColumnTransformer([
        ("numeric", numeric, settings["numeric_features"]),
        ("category", categorical, settings["categorical_features"]),
    ])
    params = candidate["params"].copy()
    if candidate["kind"] == "ridge":
        estimator = Ridge(**params)
    elif candidate["kind"] == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(random_state=settings["seed"], **params)
    elif candidate["kind"] == "random_forest":
        estimator = RandomForestRegressor(random_state=settings["seed"], n_jobs=1, **params)
    else:
        raise ValueError(f"Unknown model kind: {candidate['kind']}")
    return Pipeline([("preprocess", transform), ("regressor", estimator)])
