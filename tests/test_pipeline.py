"""Testes da pipeline de ML com dados sintéticos (rodam no CI, sem o data lake).

Garantem: ausência de variáveis de vazamento, split sem municípios compartilhados,
pré-processamento integrado ao modelo e saída de probabilidades válida.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import best_threshold, classification_report_dict
from src.modeling import train
from src.modeling.pipelines import MODEL_INPUT, make_pipeline
from src.preprocessing.features import CATEGORICAL_FEATURES, LEAKAGE_COLUMNS, NUMERIC_FEATURES, TARGET
from src.preprocessing.transformers import ENGINEERED_FEATURES, FeatureEngineer


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(0)
    n, n_mun = 6000, 300
    df = pd.DataFrame({c: rng.normal(50, 15, n) for c in NUMERIC_FEATURES})
    df["capital"] = rng.integers(0, 2, n)
    df["populacao_2019"] = rng.integers(1_000, 1_000_000, n)
    df["pib_per_capita_2010"] = rng.uniform(2_000, 60_000, n)
    df["esc_n_matriculados"] = rng.integers(5, 300, n)
    df["rede"] = rng.choice(["Municipal", "Estadual"], n)
    df["sigla_uf"] = rng.choice(["SP", "CE", "BA", "RS"], n)
    df["regiao"] = rng.choice(["Sudeste", "Nordeste", "Sul"], n)
    df["porte_municipio"] = rng.choice(["até 10 mil", "10–50 mil", "500 mil+"], n)
    df.loc[rng.random(n) < 0.2, "esc_pct_alf_ant"] = np.nan  # ausências estruturais
    df["id_municipio"] = rng.integers(0, n_mun, n).astype(str)
    df["id_escola"] = rng.integers(0, 1500, n)
    logit = 0.06 * (df["mun_taxa_rede_ant"] - 50) + rng.normal(0, 1, n)
    df[TARGET] = (logit > 0).astype(int)
    return df


def test_sem_variaveis_de_vazamento():
    assert not set(LEAKAGE_COLUMNS) & set(MODEL_INPUT)
    assert "proficiencia" not in MODEL_INPUT
    assert set(CATEGORICAL_FEATURES) <= set(MODEL_INPUT)


def test_split_agrupado_sem_sobreposicao(synthetic):
    s = train.split_by_group(synthetic)
    assert not set(s.train.id_municipio) & set(s.test.id_municipio)
    assert not set(s.train.id_municipio) & set(s.val.id_municipio)
    assert len(s.train) + len(s.val) + len(s.test) == len(synthetic)


def test_feature_engineer_cria_atributos(synthetic):
    out = FeatureEngineer().fit_transform(synthetic[MODEL_INPUT])
    assert set(ENGINEERED_FEATURES) <= set(out.columns)
    assert out["esc_sem_historico"].isin([0, 1]).all()


@pytest.mark.parametrize("nome,params", [("logistic", {}), ("lightgbm", {"n_estimators": 30})])
def test_pipeline_treina_e_preve(synthetic, nome, params):
    s = train.split_by_group(synthetic)
    pipe = make_pipeline(nome, **params).fit(s.train[MODEL_INPUT], s.train[TARGET])
    proba = pipe.predict_proba(s.test[MODEL_INPUT])[:, 1]
    assert proba.shape == (len(s.test),)
    assert np.all((proba >= 0) & (proba <= 1))
    m = classification_report_dict(s.test[TARGET], proba, best_threshold(s.val[TARGET], pipe.predict_proba(s.val[MODEL_INPUT])[:, 1]))
    assert m["roc_auc"] > 0.7  # o sinal sintético deve ser aprendido


def test_categoria_desconhecida_nao_quebra(synthetic):
    pipe = make_pipeline("logistic").fit(synthetic[MODEL_INPUT], synthetic[TARGET])
    novo = synthetic[MODEL_INPUT].head(5).copy()
    novo["sigla_uf"] = "XX"
    assert pipe.predict_proba(novo).shape == (5, 2)
