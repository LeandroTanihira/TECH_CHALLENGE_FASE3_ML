"""Modelos de aplicação estratégica (nível município).

1. ``risk_model``   — classificador supervisionado: o município vai NÃO atingir
   a meta do ano seguinte? Treinado com contexto 2023 → resultado 2024 e
   aplicado com contexto 2024 → risco para a meta de 2025.
2. ``clustering``   — KMeans sobre perfis municipais: quais municípios/regiões
   têm padrões educacionais e socioeconômicos semelhantes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE
from src.preprocessing.features import MUNI_CATEGORICAL, MUNI_NUMERIC

RISK_TARGET = "nao_atingiu_meta"


def _prep():
    num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, MUNI_NUMERIC), ("cat", cat, MUNI_CATEGORICAL)],
                             verbose_feature_names_out=False)


def make_risk_pipeline(kind="random_forest") -> Pipeline:
    if kind == "logistic":
        model = LogisticRegression(max_iter=3000, C=0.5, random_state=RANDOM_STATE)
    else:
        model = RandomForestClassifier(n_estimators=500, min_samples_leaf=25, max_features="sqrt",
                                       n_jobs=-1, random_state=RANDOM_STATE)
    return Pipeline([("preprocess", _prep()), ("model", model)])


def evaluate_risk_models(train: pd.DataFrame) -> pd.DataFrame:
    d = train.dropna(subset=[RISK_TARGET])
    X, y = d[MUNI_NUMERIC + MUNI_CATEGORICAL], d[RISK_TARGET].astype(int)
    rows = []
    esquemas = (
        ("estratificada", StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), None),
        # Mais rigorosa: o modelo precisa generalizar para UFs que não viu no treino
        ("agrupada por UF", GroupKFold(n_splits=5), d["sigla_uf"]),
    )
    for kind in ("logistic", "random_forest"):
        for nome_cv, cv, groups in esquemas:
            r = cross_validate(make_risk_pipeline(kind), X, y, cv=cv, groups=groups, return_train_score=True,
                               scoring=["roc_auc", "balanced_accuracy", "f1", "recall", "precision"])
            rows.append({"modelo": kind, "validacao": nome_cv,
                         **{f"{m}_cv": r[f"test_{m}"].mean() for m in ("roc_auc", "balanced_accuracy", "f1", "recall", "precision")},
                         "roc_auc_cv_desvio": r["test_roc_auc"].std(),
                         "roc_auc_treino": r["train_roc_auc"].mean()})
    return pd.DataFrame(rows)


def fit_risk_model(train: pd.DataFrame, kind="random_forest") -> Pipeline:
    d = train.dropna(subset=[RISK_TARGET])
    pipe = make_risk_pipeline(kind)
    pipe.fit(d[MUNI_NUMERIC + MUNI_CATEGORICAL], d[RISK_TARGET].astype(int))
    return pipe


def score_risk(pipe, df: pd.DataFrame) -> pd.Series:
    return pd.Series(pipe.predict_proba(df[MUNI_NUMERIC + MUNI_CATEGORICAL])[:, 1], index=df.index)


def risk_band(p: pd.Series) -> pd.Series:
    return pd.cut(p, [-0.01, 0.4, 0.6, 0.8, 1.0], labels=["Baixo", "Moderado", "Alto", "Crítico"])


# --------------------------------------------------------------------------
# Clusterização (não supervisionado)
# --------------------------------------------------------------------------
CLUSTER_FEATURES = [
    "taxa_2024", "evolucao_23_24", "participacao_2024", "gap_meta_2030",
    "log_populacao", "log_pib_per_capita", "dist_capital_km",
]


def clustering_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    d = df.dropna(subset=["taxa_2024"]).copy()
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    Z = pipe.fit_transform(d[CLUSTER_FEATURES])
    return d, Z


def choose_k(Z: np.ndarray, ks=range(2, 11)) -> pd.DataFrame:
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(Z)
        sil = silhouette_score(Z, km.labels_, sample_size=min(4000, len(Z)), random_state=RANDOM_STATE)
        rows.append({"k": k, "inercia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def fit_clusters(Z: np.ndarray, k: int):
    km = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE).fit(Z)
    pcs = PCA(n_components=2, random_state=RANDOM_STATE).fit(Z)
    return km, pcs
