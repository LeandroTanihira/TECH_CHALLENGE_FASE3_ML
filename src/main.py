"""Orquestrador da pipeline de ML — reproduz todo o projeto de ponta a ponta.

Uso:
    python -m src.main            # executa todas as etapas
    python -m src.main --fast     # amostras menores (teste rápido)

Cada etapa grava seus artefatos em ``models/``, ``reports/`` e ``images/``
e pode ser chamada isoladamente pelos notebooks.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src import config
from src.evaluation import interpretability as interp
from src.evaluation.metrics import best_threshold, classification_report_dict, metrics_by_group
from src.modeling import strategic, train
from src.modeling.pipelines import MODEL_INPUT
from src.preprocessing.features import TARGET, build_municipal_dataset, build_municipal_profile, build_student_dataset

STUDENTS_CACHE = config.PROCESSED_DIR / "alunos_modelagem_2024.parquet"
METRICS_JSON = config.REPORTS_DIR / "metricas.json"


def _update_metrics(key: str, value):
    data = json.loads(METRICS_JSON.read_text()) if METRICS_JSON.exists() else {}
    data[key] = value
    METRICS_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=float))


# ------------------------------------------------------------------ etapa 1
def step_dataset(force=False) -> pd.DataFrame:
    if STUDENTS_CACHE.exists() and not force:
        return pd.read_parquet(STUDENTS_CACHE)
    df = build_student_dataset(config.TARGET_YEAR, config.CONTEXT_YEAR)
    df.to_parquet(STUDENTS_CACHE, index=False)
    return df


def step_split(df) -> train.Splits:
    s = train.split_by_group(df)
    s.summary().to_csv(config.REPORTS_DIR / "splits.csv", index=False)
    return s


# ------------------------------------------------------------------ etapa 2
def step_compare(s: train.Splits, sample_n=200_000, n_splits=5) -> pd.DataFrame:
    res = train.compare_models(s.train, sample_n=sample_n, n_splits=n_splits)
    res.to_csv(config.REPORTS_DIR / "comparacao_modelos_cv.csv", index=False)
    return res


# ------------------------------------------------------------------ etapa 3
def step_tune(s: train.Splits, sample_n=300_000, n_iter=20) -> dict:
    search = train.tune_lightgbm(s.train, sample_n=sample_n, n_iter=n_iter)
    cv = pd.DataFrame(search.cv_results_)
    keep = [c for c in cv.columns if c.startswith("param_")] + ["mean_test_score", "std_test_score", "mean_train_score", "rank_test_score"]
    cv[keep].sort_values("rank_test_score").to_csv(config.REPORTS_DIR / "tuning_lightgbm.csv", index=False)
    best = {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
    _update_metrics("tuning", {"best_params": best, "best_auc_cv": search.best_score_,
                               "auc_treino_best": cv.loc[search.best_index_, "mean_train_score"]})
    return best


# ------------------------------------------------------------------ etapa 4
def step_final(s: train.Splits, best_params: dict) -> dict:
    t = time.time()
    pipe = train.fit_pipeline(s.train, "lightgbm", **best_params)
    p_tr = train.predict_proba(pipe, s.train)
    p_va = train.predict_proba(pipe, s.val)
    p_te = train.predict_proba(pipe, s.test)
    thr = best_threshold(s.val[TARGET], p_va, "f1_macro")  # limiar escolhido na validação
    out = {
        "treino": classification_report_dict(s.train[TARGET], p_tr, thr),
        "validacao": classification_report_dict(s.val[TARGET], p_va, thr),
        "teste": classification_report_dict(s.test[TARGET], p_te, thr),
        "threshold": thr, "tempo_treino_s": time.time() - t,
    }
    # Baseline logístico no mesmo protocolo (referência interpretável)
    logit = train.fit_pipeline(s.train, "logistic")
    out["teste_logistic"] = classification_report_dict(s.test[TARGET], train.predict_proba(logit, s.test), 0.5)
    _update_metrics("modelo_final_alunos", out)
    train.save_model(pipe, "modelo_alfabetizacao_lgbm", {"params": best_params, "threshold": thr,
                                                         "features": MODEL_INPUT, "treino_ano_ctx": config.CONTEXT_YEAR,
                                                         "treino_ano_alvo": config.TARGET_YEAR})
    train.save_model(logit, "modelo_alfabetizacao_logistic")

    pred = s.test[["id_aluno", "id_escola", "id_municipio", "sigla_uf", "regiao", "rede", TARGET]].copy()
    pred["proba"] = p_te
    pred.to_parquet(config.PROCESSED_DIR / "predicoes_teste.parquet", index=False)
    for g in ("regiao", "rede"):
        metrics_by_group(pred.rename(columns={TARGET: "y"}), g, threshold=thr).to_csv(
            config.REPORTS_DIR / f"metricas_por_{g}.csv", index=False)
    return out


# ------------------------------------------------------------------ etapa 5
def step_interpret(s: train.Splits, n_perm=60_000, n_shap=5000):
    pipe = train.load_model("modelo_alfabetizacao_lgbm")
    interp.model_importance(pipe).to_csv(config.REPORTS_DIR / "importancia_ganho.csv", index=False)
    te = s.test.sample(n=min(n_perm, len(s.test)), random_state=config.RANDOM_STATE)
    perm = interp.permutation_importance_df(pipe, te[MODEL_INPUT], te[TARGET])
    perm.to_csv(config.REPORTS_DIR / "importancia_permutacao.csv", index=False)
    sv, Xt = interp.shap_values(pipe, s.test[MODEL_INPUT], max_rows=n_shap)
    interp.shap_summary(sv).to_csv(config.REPORTS_DIR / "shap_importancia.csv", index=False)
    return pipe, perm, sv, Xt


# ------------------------------------------------------------------ etapa 6
def step_municipal_validation(s: train.Splits) -> pd.DataFrame:
    """Agrega as probabilidades dos alunos de TESTE por município e compara
    com a taxa real 2024 — o modelo individual vira um termômetro municipal."""
    pred = pd.read_parquet(config.PROCESSED_DIR / "predicoes_teste.parquet")
    agg = pred.groupby(["id_municipio", "sigla_uf", "regiao"], as_index=False).agg(
        alunos=("proba", "size"), taxa_real=(TARGET, "mean"), taxa_prevista=("proba", "mean"))
    agg[["taxa_real", "taxa_prevista"]] *= 100
    agg.to_csv(config.REPORTS_DIR / "validacao_municipal_teste.csv", index=False)
    # Baseline ingênuo: "o município repete a taxa do ano anterior" (rede municipal)
    ctx = s.test[s.test["rede"] == "Municipal"].drop_duplicates("id_municipio")[["id_municipio", "mun_taxa_rede_ant"]]
    agg = agg.merge(ctx.rename(columns={"mun_taxa_rede_ant": "taxa_ano_anterior"}), on="id_municipio", how="left")
    agg.to_csv(config.REPORTS_DIR / "validacao_municipal_teste.csv", index=False)
    big = agg[agg["alunos"] >= 30]
    err = big["taxa_prevista"] - big["taxa_real"]
    b = big.dropna(subset=["taxa_ano_anterior"])
    _update_metrics("agregacao_municipal_teste", {
        "municipios": int(len(big)), "correlacao": big[["taxa_real", "taxa_prevista"]].corr().iloc[0, 1],
        "mae_pp": err.abs().mean(), "vies_pp": err.mean(),
        "baseline_ano_anterior": {
            "municipios": int(len(b)),
            "correlacao": b[["taxa_real", "taxa_ano_anterior"]].corr().iloc[0, 1],
            "mae_pp": (b["taxa_ano_anterior"] - b["taxa_real"]).abs().mean(),
            "mae_modelo_mesmos_municipios_pp": (b["taxa_prevista"] - b["taxa_real"]).abs().mean(),
        }})
    return agg


def step_strategic(k=None) -> dict:
    # (a) risco de não atingir a meta do ano seguinte
    tr = build_municipal_dataset(2023, 2024)
    sc = build_municipal_dataset(2024, 2025)
    cv = strategic.evaluate_risk_models(tr)
    cv.to_csv(config.REPORTS_DIR / "risco_meta_cv.csv", index=False)
    # Regressão logística: AUC um pouco menor na CV estratificada, porém muito mais robusta
    # na CV agrupada por UF (generaliza para contextos não vistos e não "decora" choques
    # estaduais pontuais, como as enchentes do RS em 2024) — ver notebook 04.
    pipe = strategic.fit_risk_model(tr, "logistic")
    train.save_model(pipe, "modelo_risco_meta_municipal")
    sc["prob_nao_atingir_meta_2025"] = strategic.score_risk(pipe, sc)
    # Sem meta 2025 pactuada (ou sem resultado 2024) o alvo não é definido → não ranquear
    sem_alvo = sc["meta_ano_alvo"].isna() | sc["taxa_ant"].isna()
    sc.loc[sem_alvo, "prob_nao_atingir_meta_2025"] = np.nan
    sc["faixa_risco"] = strategic.risk_band(sc["prob_nao_atingir_meta_2025"]).astype(object)
    sc.loc[sem_alvo, "faixa_risco"] = "Sem meta/resultado"
    cols = ["id_municipio", "nome_municipio", "sigla_uf", "regiao", "porte_municipio", "taxa_ant",
            "meta_ano_alvo", "esforco_necessario", "participacao_ant", "prob_nao_atingir_meta_2025", "faixa_risco"]
    ranking = sc[cols].rename(columns={"taxa_ant": "taxa_2024", "meta_ano_alvo": "meta_2025",
                                       "participacao_ant": "participacao_2024"})
    ranking = ranking.sort_values("prob_nao_atingir_meta_2025", ascending=False)
    ranking.to_csv(config.REPORTS_DIR / "ranking_risco_municipios_2025.csv", index=False)

    coef = pipe.named_steps["model"].coef_[0]
    imp = pd.DataFrame({"feature": pipe.named_steps["preprocess"].get_feature_names_out(), "coeficiente": coef,
                        "odds_ratio_por_dp": np.exp(coef)})
    imp = imp.reindex(imp["coeficiente"].abs().sort_values(ascending=False).index)
    imp.to_csv(config.REPORTS_DIR / "risco_meta_importancia.csv", index=False)

    # (b) clusterização de perfis municipais
    prof = build_municipal_profile()
    d, Z = strategic.clustering_matrix(prof)
    ks = strategic.choose_k(Z)
    ks.to_csv(config.REPORTS_DIR / "clusters_escolha_k.csv", index=False)
    k = k or 4
    km, pca = strategic.fit_clusters(Z, k)
    d["cluster"] = km.labels_
    pcs = pca.transform(Z)
    d["pc1"], d["pc2"] = pcs[:, 0], pcs[:, 1]
    d = d.merge(ranking[["id_municipio", "prob_nao_atingir_meta_2025", "faixa_risco"]], on="id_municipio", how="left")
    d.to_csv(config.REPORTS_DIR / "clusters_municipios.csv", index=False)

    summary = {"risco_cv": cv.to_dict(orient="records"),
               "faixas_risco_2025": ranking["faixa_risco"].value_counts().to_dict(),
               "k_clusters": k, "silhouette": ks.set_index("k").loc[k, "silhouette"],
               "variancia_pca_2d": float(pca.explained_variance_ratio_.sum())}
    _update_metrics("estrategico", summary)
    return summary


def run_all(fast=False):
    t0 = time.time()
    print("1/6 Base de modelagem (Gold → alunos 2024 com contexto 2023)")
    df = step_dataset()
    s = step_split(df)
    print(s.summary().to_string(index=False))
    print("2/6 Comparação de modelos (GroupKFold)")
    print(step_compare(s, sample_n=60_000 if fast else 200_000, n_splits=3 if fast else 5).round(4).to_string(index=False))
    print("3/6 Otimização de hiperparâmetros (RandomizedSearchCV)")
    best = step_tune(s, sample_n=60_000 if fast else 300_000, n_iter=4 if fast else 20)
    print(best)
    print("4/6 Treino final + limiar na validação + avaliação no teste")
    out = step_final(s, best)
    print({k: round(v, 4) for k, v in out["teste"].items() if isinstance(v, float)})
    print("5/6 Interpretabilidade (permutação + SHAP)")
    step_interpret(s, n_perm=20_000 if fast else 60_000, n_shap=2000 if fast else 5000)
    step_municipal_validation(s)
    print("6/6 Aplicação estratégica (risco de meta + clusters)")
    print(step_strategic())
    print(f"Concluído em {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    run_all(ap.parse_args().fast)
