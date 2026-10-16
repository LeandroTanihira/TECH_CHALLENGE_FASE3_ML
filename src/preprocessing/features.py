"""Construção das bases analíticas (aluno e município) a partir da Gold.

Desenho anti-leakage (ver README §5):

* **Alvo** — ``alfabetizado`` do aluno no ano-alvo (T). Apenas alunos
  **presentes** entram: para ausentes o alvo não é observado (a Silver grava 0,
  o que seria um falso "não alfabetizado").
* **Features de contexto são sempre do ano anterior (T-1)** — taxa da escola,
  do município e da UF em T-1. Taxas de T conteriam o próprio resultado do aluno
  (vazamento direto do alvo).
* **Variáveis pós-prova são descartadas** — ``proficiencia`` (define o alvo:
  ≥ 743), ``preenchimento_caderno`` e ``peso_aluno`` (calculados após a aplicação).
* Metas (definidas antes da prova), características territoriais e o
  porte da escola em T (matrícula, conhecido antes da prova) são permitidos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.preprocessing.external import build_external_features
from src.preprocessing.load import load_alunos, load_gold

LEAKAGE_COLUMNS = ["proficiencia", "preenchimento_caderno", "peso_aluno", "presenca"]
ID_COLUMNS = ["id_aluno", "id_escola", "id_municipio", "ano"]
TARGET = "alfabetizado"

CATEGORICAL_FEATURES = ["rede", "sigla_uf", "regiao", "porte_municipio"]
NUMERIC_FEATURES = [
    # contexto da escola (T-1)
    "esc_pct_alf_ant", "esc_prof_media_ant", "esc_taxa_presenca_ant", "esc_n_alunos_ant",
    # porte da escola em T (matrícula — conhecida antes da prova)
    "esc_n_matriculados",
    # contexto municipal (T-1)
    "mun_taxa_rede_ant", "mun_taxa_publica_ant", "mun_media_port_ant", "mun_participacao_ant",
    # metas (definidas a priori)
    "mun_meta_ano_alvo", "uf_taxa_ant", "uf_meta_ano_alvo",
    # território / socioeconômico (IBGE)
    "capital", "dist_capital_km", "populacao_2019", "cresc_pop_2012_2019_pct",
    "pib_per_capita_2010", "latitude", "longitude",
]


# --------------------------------------------------------------------------
# Contextos (sempre calculados sobre o ano de contexto, T-1)
# --------------------------------------------------------------------------
def school_context(ano: int) -> pd.DataFrame:
    """Indicadores da escola no ano ``ano`` (usados como histórico em ano+1)."""
    a = load_alunos(ano, columns=["id_escola", "presenca", "alfabetizado", "proficiencia", "peso_aluno"])
    pres = a[a["presenca"] == 1]
    g_pres = pres.groupby("id_escola").agg(
        esc_pct_alf_ant=("alfabetizado", "mean"),
        esc_prof_media_ant=("proficiencia", "mean"),
        esc_n_alunos_ant=("alfabetizado", "size"),
    )
    g_all = a.groupby("id_escola").agg(esc_taxa_presenca_ant=("presenca", "mean"))
    out = g_pres.join(g_all, how="outer").reset_index()
    out["esc_pct_alf_ant"] *= 100
    out["esc_taxa_presenca_ant"] *= 100
    return out


def municipal_context(ano: int) -> pd.DataFrame:
    """Indicadores municipais (Gold) do ano ``ano``, por município e rede."""
    ind = load_gold("indicador_municipio")
    ind = ind[(ind["ano"] == ano) & (ind["origem"] == "batch")]

    por_rede = ind[ind["rede"].isin(["Municipal", "Estadual"])][["id_municipio", "rede", "taxa_alfabetizacao"]]
    por_rede = por_rede.rename(columns={"taxa_alfabetizacao": "mun_taxa_rede_ant"})

    pub = ind[ind["rede"] == "Pública"][["id_municipio", "taxa_alfabetizacao", "media_portugues"]]
    pub = pub.rename(columns={"taxa_alfabetizacao": "mun_taxa_publica_ant", "media_portugues": "mun_media_port_ant"})

    perfil = load_gold("perfil_alunos_municipio")
    perfil = perfil[perfil["ano"] == ano][["id_municipio", "rede", "taxa_participacao"]]
    perfil = perfil.rename(columns={"taxa_participacao": "mun_participacao_ant"})

    out = por_rede.merge(perfil, on=["id_municipio", "rede"], how="outer")
    return out, pub


def metas_municipio(ano_alvo: int) -> pd.DataFrame:
    """Meta municipal (rede municipal) para o ano-alvo e para 2030."""
    ind = load_gold("indicador_municipio")
    m = ind[(ind["rede"] == "Municipal") & (ind["origem"] == "batch")].sort_values("ano")
    m = m.drop_duplicates("id_municipio", keep="last")
    col = f"meta_alfabetizacao_{ano_alvo}"
    return m[["id_municipio", col, "meta_alfabetizacao_2030"]].rename(
        columns={col: "mun_meta_ano_alvo", "meta_alfabetizacao_2030": "mun_meta_2030"}
    )


def uf_context(ano_ctx: int, ano_alvo: int) -> pd.DataFrame:
    uf = load_gold("metas_vs_resultados_uf")
    taxa = uf[uf["ano"] == ano_ctx][["sigla_uf", "taxa_alfabetizacao"]].rename(columns={"taxa_alfabetizacao": "uf_taxa_ant"})
    meta = uf.drop_duplicates("sigla_uf", keep="last")[["sigla_uf", f"meta_alfabetizacao_{ano_alvo}"]]
    meta = meta.rename(columns={f"meta_alfabetizacao_{ano_alvo}": "uf_meta_ano_alvo"})
    return taxa.merge(meta, on="sigla_uf", how="outer")


def porte_municipio(pop: pd.Series) -> pd.Series:
    bins = [0, 10_000, 50_000, 100_000, 500_000, np.inf]
    labels = ["até 10 mil", "10–50 mil", "50–100 mil", "100–500 mil", "500 mil+"]
    return pd.cut(pop, bins=bins, labels=labels).astype(str).replace("nan", np.nan)


# --------------------------------------------------------------------------
# Base de alunos (nível de modelagem principal)
# --------------------------------------------------------------------------
def build_student_dataset(
    ano_alvo: int = config.TARGET_YEAR,
    ano_ctx: int = config.CONTEXT_YEAR,
    sample_frac: float | None = None,
    only_present: bool = True,
) -> pd.DataFrame:
    """Uma linha por aluno do ano-alvo com features de contexto de ``ano_ctx``."""
    alunos = load_alunos(ano_alvo, columns=["id_aluno", "id_escola", "id_municipio", "rede", "presenca", "alfabetizado", "sigla_uf"])
    # Porte atual da escola (matrícula) — calculado ANTES de filtrar presentes
    alunos["esc_n_matriculados"] = alunos.groupby("id_escola")["id_aluno"].transform("size")
    if only_present:
        alunos = alunos[alunos["presenca"] == 1]
    if sample_frac:
        alunos = alunos.sample(frac=sample_frac, random_state=config.RANDOM_STATE)

    df = alunos.merge(school_context(ano_ctx), on="id_escola", how="left")
    mun_rede, mun_pub = municipal_context(ano_ctx)
    df = df.merge(mun_rede, on=["id_municipio", "rede"], how="left")
    df = df.merge(mun_pub, on="id_municipio", how="left")
    df = df.merge(metas_municipio(ano_alvo), on="id_municipio", how="left")
    df = df.merge(uf_context(ano_ctx, ano_alvo), on="sigla_uf", how="left")

    ext = build_external_features().drop(columns=["sigla_uf", "nome_municipio"])
    df = df.merge(ext, on="id_municipio", how="left")
    df["porte_municipio"] = porte_municipio(df["populacao_2019"])
    df["ano"] = ano_alvo
    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Base de municípios (risco de não atingir a meta / clusterização)
# --------------------------------------------------------------------------
MUNI_NUMERIC = [
    "taxa_ant", "media_port_ant", "participacao_ant", "prof_media_ant",
    "meta_ano_alvo", "esforco_necessario", "uf_taxa_ant",
    "capital", "dist_capital_km", "populacao_2019", "cresc_pop_2012_2019_pct",
    "pib_per_capita_2010",
]
MUNI_CATEGORICAL = ["regiao", "porte_municipio"]


def build_municipal_dataset(ano_ctx: int, ano_alvo: int) -> pd.DataFrame:
    """Rede municipal: features de ``ano_ctx`` e (quando existir) o alvo em ``ano_alvo``.

    Alvo: ``nao_atingiu_meta`` = taxa(ano_alvo) < meta(ano_alvo).
    """
    ind = load_gold("indicador_municipio")
    ind = ind[(ind["rede"] == "Municipal") & (ind["origem"] == "batch")]
    ctx = ind[ind["ano"] == ano_ctx][["id_municipio", "sigla_uf", "taxa_alfabetizacao", "media_portugues"]]
    ctx = ctx.rename(columns={"taxa_alfabetizacao": "taxa_ant", "media_portugues": "media_port_ant"})

    perfil = load_gold("perfil_alunos_municipio")
    perfil = perfil[(perfil["ano"] == ano_ctx) & (perfil["rede"] == "Municipal")]
    perfil = perfil[["id_municipio", "taxa_participacao", "proficiencia_media", "alunos_matriculados"]].rename(
        columns={"taxa_participacao": "participacao_ant", "proficiencia_media": "prof_media_ant",
                 "alunos_matriculados": "matriculados_ant"}
    )
    df = ctx.merge(perfil, on="id_municipio", how="left")

    metas = ind.sort_values("ano").drop_duplicates("id_municipio", keep="last")
    col = f"meta_alfabetizacao_{ano_alvo}"
    df = df.merge(metas[["id_municipio", col, "meta_alfabetizacao_2030"]].rename(
        columns={col: "meta_ano_alvo", "meta_alfabetizacao_2030": "meta_2030"}), on="id_municipio", how="left")
    df["esforco_necessario"] = df["meta_ano_alvo"] - df["taxa_ant"]

    df = df.merge(uf_context(ano_ctx, ano_alvo)[["sigla_uf", "uf_taxa_ant"]], on="sigla_uf", how="left")
    ext = build_external_features().drop(columns=["sigla_uf"])
    df = df.merge(ext, on="id_municipio", how="left")
    df["porte_municipio"] = porte_municipio(df["populacao_2019"])

    alvo = ind[ind["ano"] == ano_alvo][["id_municipio", "taxa_alfabetizacao"]].rename(
        columns={"taxa_alfabetizacao": "taxa_alvo"})
    df = df.merge(alvo, on="id_municipio", how="left")
    df["nao_atingiu_meta"] = np.where(
        df["taxa_alvo"].notna() & df["meta_ano_alvo"].notna(),
        (df["taxa_alvo"] < df["meta_ano_alvo"]).astype(float), np.nan,
    )
    df["ano_ctx"], df["ano_alvo"] = ano_ctx, ano_alvo
    return df


def build_municipal_profile() -> pd.DataFrame:
    """Perfil 2024 da rede municipal (para clusterização e painel de risco)."""
    ind = load_gold("indicador_municipio")
    ind = ind[(ind["rede"] == "Municipal") & (ind["origem"] == "batch")]
    w = ind.pivot_table(index="id_municipio", columns="ano", values="taxa_alfabetizacao")
    prof = pd.DataFrame({"taxa_2023": w.get(2023), "taxa_2024": w.get(2024)})
    prof["evolucao_23_24"] = prof["taxa_2024"] - prof["taxa_2023"]
    metas = ind.sort_values("ano").drop_duplicates("id_municipio", keep="last").set_index("id_municipio")
    prof["meta_2025"] = metas["meta_alfabetizacao_2025"]
    prof["meta_2030"] = metas["meta_alfabetizacao_2030"]
    prof["gap_meta_2030"] = prof["meta_2030"] - prof["taxa_2024"]
    perfil = load_gold("perfil_alunos_municipio")
    perfil = perfil[(perfil["ano"] == 2024) & (perfil["rede"] == "Municipal")].set_index("id_municipio")
    prof["participacao_2024"] = perfil["taxa_participacao"]
    prof["matriculados_2024"] = perfil["alunos_matriculados"]
    prof = prof.reset_index()
    ext = build_external_features()
    prof = prof.merge(ext, on="id_municipio", how="left")
    prof["log_populacao"] = np.log1p(prof["populacao_2019"])
    prof["log_pib_per_capita"] = np.log1p(prof["pib_per_capita_2010"])
    prof["porte_municipio"] = porte_municipio(prof["populacao_2019"])
    return prof
