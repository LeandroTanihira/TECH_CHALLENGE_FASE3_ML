"""Fábrica de pipelines de ML (pré-processamento integrado ao estimador)."""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.preprocessing.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.preprocessing.transformers import ENGINEERED_FEATURES, FeatureEngineer, build_preprocessor

ALL_NUMERIC = NUMERIC_FEATURES + ENGINEERED_FEATURES
MODEL_INPUT = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_estimator(name: str, **params):
    if name == "baseline":
        return DummyClassifier(strategy="prior")
    if name == "logistic":
        return LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE, **params)
    if name == "random_forest":
        defaults = dict(n_estimators=200, max_depth=14, min_samples_leaf=100, max_features="sqrt",
                        n_jobs=-1, random_state=RANDOM_STATE)
        return RandomForestClassifier(**{**defaults, **params})
    if name == "hist_gb":
        defaults = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=200,
                        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                        random_state=RANDOM_STATE)
        return HistGradientBoostingClassifier(**{**defaults, **params})
    if name == "lightgbm":
        defaults = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        return LGBMClassifier(**{**defaults, **params})
    raise ValueError(f"Modelo desconhecido: {name}")


def make_pipeline(name: str, numeric=None, categorical=None, engineer: bool = True, **params) -> Pipeline:
    """Pipeline completo: engenharia de atributos → imputação/encoding/escala → modelo."""
    numeric = list(numeric or (ALL_NUMERIC if engineer else NUMERIC_FEATURES))
    categorical = list(categorical or CATEGORICAL_FEATURES)
    kind = "linear" if name in ("logistic", "baseline") else "tree"
    steps = []
    if engineer:
        steps.append(("features", FeatureEngineer()))
    steps += [
        ("preprocess", build_preprocessor(numeric, categorical, kind=kind)),
        ("model", make_estimator(name, **params)),
    ]
    return Pipeline(steps)
