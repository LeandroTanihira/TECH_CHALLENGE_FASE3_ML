"""Treinamento, validação cruzada, otimização e persistência do modelo de alunos."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV, cross_validate

from src import config
from src.modeling.pipelines import MODEL_INPUT, make_pipeline
from src.preprocessing.features import TARGET

GROUP = "id_municipio"


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, d in (("treino", self.train), ("validação", self.val), ("teste", self.test)):
            rows.append({"conjunto": name, "alunos": len(d), "municipios": d[GROUP].nunique(),
                         "escolas": d["id_escola"].nunique(), "taxa_alfabetizados": d[TARGET].mean()})
        return pd.DataFrame(rows)


def split_by_group(df: pd.DataFrame, val_size=0.15, test_size=0.15, seed=config.RANDOM_STATE) -> Splits:
    """Treino/validação/teste **agrupados por município**.

    Alunos do mesmo município (e, portanto, da mesma escola) nunca aparecem em
    dois conjuntos — evita que o modelo "decore" o contexto local e superestime
    a performance (vazamento por agrupamento).
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size + test_size, random_state=seed)
    tr_idx, tmp_idx = next(gss.split(df, groups=df[GROUP]))
    tmp = df.iloc[tmp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_size / (val_size + test_size), random_state=seed)
    va_idx, te_idx = next(gss2.split(tmp, groups=tmp[GROUP]))
    s = Splits(df.iloc[tr_idx].copy(), tmp.iloc[va_idx].copy(), tmp.iloc[te_idx].copy())
    assert not set(s.train[GROUP]) & set(s.test[GROUP]), "vazamento de municípios entre treino e teste"
    assert not set(s.train[GROUP]) & set(s.val[GROUP]), "vazamento de municípios entre treino e validação"
    return s


def _sample(df, n, seed=config.RANDOM_STATE):
    return df.sample(n=min(n, len(df)), random_state=seed) if n else df


def compare_models(train: pd.DataFrame, models=("baseline", "logistic", "random_forest", "hist_gb", "lightgbm"),
                   sample_n=200_000, n_splits=5) -> pd.DataFrame:
    """Validação cruzada agrupada (GroupKFold) — compara famílias de modelos."""
    d = _sample(train, sample_n)
    X, y, g = d[MODEL_INPUT], d[TARGET], d[GROUP]
    cv = GroupKFold(n_splits=n_splits)
    rows = []
    for name in models:
        t = time.time()
        res = cross_validate(make_pipeline(name), X, y, groups=g, cv=cv,
                             scoring=["roc_auc", "balanced_accuracy", "neg_brier_score"],
                             return_train_score=True, n_jobs=1)
        rows.append({
            "modelo": name,
            "auc_cv_media": res["test_roc_auc"].mean(), "auc_cv_desvio": res["test_roc_auc"].std(),
            "auc_treino": res["train_roc_auc"].mean(),
            "gap_overfitting": res["train_roc_auc"].mean() - res["test_roc_auc"].mean(),
            "bal_acc_cv": res["test_balanced_accuracy"].mean(),
            "brier_cv": -res["test_neg_brier_score"].mean(),
            "tempo_s": time.time() - t,
        })
        print(f"  {name:14s} AUC={rows[-1]['auc_cv_media']:.4f} ± {rows[-1]['auc_cv_desvio']:.4f}")
    return pd.DataFrame(rows).sort_values("auc_cv_media", ascending=False).reset_index(drop=True)


LGBM_SPACE = {
    "model__n_estimators": randint(200, 800),
    "model__learning_rate": loguniform(0.02, 0.15),
    "model__num_leaves": randint(15, 127),
    "model__min_child_samples": randint(50, 1000),
    "model__subsample": uniform(0.6, 0.4),
    "model__colsample_bytree": uniform(0.5, 0.5),
    "model__reg_lambda": loguniform(0.1, 30),
    "model__reg_alpha": loguniform(1e-3, 10),
}


def tune_lightgbm(train: pd.DataFrame, sample_n=300_000, n_iter=25, n_splits=3) -> RandomizedSearchCV:
    """Busca aleatória de hiperparâmetros com CV agrupada (regularização incluída
    no espaço de busca para conter overfitting)."""
    d = _sample(train, sample_n, seed=config.RANDOM_STATE + 1)
    search = RandomizedSearchCV(
        make_pipeline("lightgbm"), LGBM_SPACE, n_iter=n_iter, scoring="roc_auc",
        cv=GroupKFold(n_splits=n_splits), random_state=config.RANDOM_STATE,
        return_train_score=True, n_jobs=1, refit=False, verbose=0,
    )
    search.fit(d[MODEL_INPUT], d[TARGET], groups=d[GROUP])
    return search


def fit_pipeline(train: pd.DataFrame, name="lightgbm", **params):
    pipe = make_pipeline(name, **params)
    pipe.fit(train[MODEL_INPUT], train[TARGET])
    return pipe


def predict_proba(pipe, df: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(df[MODEL_INPUT])[:, 1]


def save_model(pipe, name: str, metadata: dict | None = None):
    path = config.MODELS_DIR / f"{name}.joblib"
    joblib.dump(pipe, path)
    if metadata is not None:
        (config.MODELS_DIR / f"{name}.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=float))
    return path


def load_model(name: str):
    return joblib.load(config.MODELS_DIR / f"{name}.joblib")
