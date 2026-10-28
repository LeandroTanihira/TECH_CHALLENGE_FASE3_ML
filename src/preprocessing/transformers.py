"""Transformadores scikit-learn: engenharia de atributos + pré-processamento.

Tudo é encapsulado no ``Pipeline`` do modelo: imputação, escala e encoding são
**ajustados apenas no treino** de cada fold (``fit``) e só aplicados
(``transform``) em validação/teste — eliminando vazamento de estatísticas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ENGINEERED_FEATURES = [
    "esc_vs_mun_ant",        # escola acima/abaixo do município em T-1
    "gap_meta_mun",          # quanto o município precisa crescer para a meta de T
    "esc_sem_historico",     # escola nova / sem alunos avaliados em T-1
    "log_populacao",
    "log_pib_per_capita",
    "log_esc_n_matriculados",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Cria atributos derivados linha a linha (sem estatísticas do conjunto,
    portanto não há aprendizado nem vazamento entre treino e teste)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["esc_vs_mun_ant"] = X["esc_pct_alf_ant"] - X["mun_taxa_rede_ant"]
        X["gap_meta_mun"] = X["mun_meta_ano_alvo"] - X["mun_taxa_rede_ant"]
        X["esc_sem_historico"] = X["esc_pct_alf_ant"].isna().astype(int)
        X["log_populacao"] = np.log1p(X["populacao_2019"])
        X["log_pib_per_capita"] = np.log1p(X["pib_per_capita_2010"])
        X["log_esc_n_matriculados"] = np.log1p(X["esc_n_matriculados"])
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(input_features or []) + ENGINEERED_FEATURES, dtype=object)


def build_preprocessor(numeric: list[str], categorical: list[str], kind: str = "linear") -> ColumnTransformer:
    """``kind='linear'``: imputação + indicador de ausência + padronização + one-hot.
    ``kind='tree'``: imputação + ordinal encoding (árvores não precisam de escala)."""
    if kind == "linear":
        num = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ])
        cat = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.005, sparse_output=False)),
        ])
    else:
        num = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True))])
        cat = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
    return ColumnTransformer(
        [("num", num, numeric), ("cat", cat, categorical)],
        remainder="drop", verbose_feature_names_out=False,
    )
