"""Adaptador para o data lake **oficial** entregue na Fase 2.

Repositório: https://github.com/postechfiap018-ctrl/TECH_CHALLENGE_FASE2_PIPELINE

A pipeline oficial roda na AWS (BigQuery → S3 Bronze → Glue Silver/Gold → Athena) e
publica:

* ``silver/resultado_municipio`` · ``silver/resultado_uf`` (rede como **código** INEP,
  particionadas por ``ano``), ``silver/metas_municipio`` · ``silver/metas_uf`` ·
  ``silver/metas_brasil`` (formato **longo**: ``ano_meta``, ``meta_valor``),
  ``silver/municipio`` · ``silver/uf`` e ``silver/alunos`` (particionada por ``ano``);
* ``gold/indicador_por_municipio`` · ``gold/comparacao_metas_resultados`` ·
  ``gold/evolucao_temporal_indicador``.

Este módulo lê essas tabelas (do S3 ou de uma cópia local obtida com
``aws s3 sync``) e materializa a **camada analítica** usada na Fase 3
(``data/analytic/``), com o mesmo contrato das funções ``load_gold``/``load_alunos``:

| Tabela analítica (Fase 3)  | Origem no lake oficial |
|---|---|
| ``indicador_municipio``     | ``silver/resultado_municipio`` + ``silver/metas_municipio`` (pivot) — validado contra ``gold/comparacao_metas_resultados`` |
| ``metas_vs_resultados_uf``  | ``silver/resultado_uf`` (rede 5) + ``silver/metas_uf`` (pivot) |
| ``evolucao_temporal``       | ``gold/evolucao_temporal_indicador`` |
| ``perfil_alunos_municipio`` | agregação ponderada de ``silver/alunos`` |
| alunos (microdados)         | ``silver/alunos`` (rede decodificada, ``sigla_uf`` derivada) |

**Atenção — amostra de alunos:** o extrator oficial usa ``TABLESAMPLE SYSTEM (2 PERCENT)``
para ``alunos`` (``QUERY_OVERRIDES`` em ``src/config.py`` do repositório da Fase 2). Para o
modelo de alunos da Fase 3 é preciso reextrair a tabela **completa** (remover o override) —
com 2% o modelo roda, mas o contexto por escola fica muito ruidoso.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from src import config

FASE2_REPO_URL = "https://github.com/postechfiap018-ctrl/TECH_CHALLENGE_FASE2_PIPELINE"

# Códigos INEP da coluna rede (tabela "dicionario" do dataset br_inep_avaliacao_alfabetizacao)
REDE_MAP = {0: "Total", 1: "Federal", 2: "Estadual", 3: "Municipal", 4: "Privada", 5: "Pública", 6: "Pública"}
META_YEARS = list(range(2024, 2031))
META_COLS = [f"meta_alfabetizacao_{a}" for a in META_YEARS]
MIN_ALUNOS_ANO_COMPLETO = 500_000  # a tabela completa tem ~1,9–2,1 milhões de alunos/ano


# ------------------------------------------------------------------ leitura
def _join(root: str, *parts: str) -> str:
    return "/".join([str(root).rstrip("/"), *parts])


def read_table(root: str, layer: str, name: str, columns: list[str] | None = None,
               filter_expr=None) -> pd.DataFrame:
    """Lê uma tabela Parquet (particionamento Hive) local ou ``s3://``."""
    dataset = ds.dataset(_join(root, layer, name), format="parquet", partitioning="hive")
    cols = None if columns is None else [c for c in columns if c in dataset.schema.names]
    return dataset.to_table(columns=cols, filter=filter_expr).to_pandas()


def is_official_layout(root) -> bool:
    root = str(root)
    if root.startswith("s3://"):
        return True  # o data lake em S3 é sempre o da pipeline oficial
    from pathlib import Path
    return (Path(root) / "gold" / "comparacao_metas_resultados").exists()


def _norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(7)


def _decode_rede(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9:
        return num.map(REDE_MAP).fillna("Desconhecida")
    return s.astype(str).str.strip().str.capitalize()


def _clip(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.where((s >= 0) & (s <= 100))


def _pivot_metas(long: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Formato longo (ano_meta, meta_valor) → largo (meta_alfabetizacao_2024..2030)."""
    long = long.copy()
    if "ano_base" in long:  # mantém a versão mais recente de cada meta
        long = long.sort_values("ano_base").drop_duplicates(keys + ["ano_meta"], keep="last")
    wide = long.pivot_table(index=keys, columns="ano_meta", values="meta_valor", aggfunc="last")
    wide.columns = [f"meta_alfabetizacao_{int(c)}" for c in wide.columns]
    for c in META_COLS:
        if c not in wide:
            wide[c] = np.nan
    return wide[META_COLS].apply(_clip).reset_index()


def _meta_do_ano(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda r: r.get(f"meta_alfabetizacao_{int(r['ano'])}", np.nan), axis=1)


# ------------------------------------------------------------------ tabelas analíticas
def build_indicador_municipio(root: str) -> pd.DataFrame:
    res = read_table(root, "silver", "resultado_municipio",
                     ["id_municipio", "ano", "rede", "serie", "taxa_alfabetizacao", "media_portugues", "sigla_uf"])
    res["id_municipio"] = _norm_id(res["id_municipio"])
    res["ano"] = res["ano"].astype(int)
    res["rede"] = _decode_rede(res["rede"])
    res["sigla_uf"] = res["id_municipio"].str[:2].astype(int).map(config.UF_IBGE)
    res["taxa_alfabetizacao"] = _clip(res["taxa_alfabetizacao"])
    res = res.dropna(subset=["taxa_alfabetizacao"])
    res["origem"] = "batch"

    metas = read_table(root, "silver", "metas_municipio")
    metas["id_municipio"] = _norm_id(metas["id_municipio"])
    wide = _pivot_metas(metas, ["id_municipio"])
    out = res.merge(wide, on="id_municipio", how="left")
    # Metas municipais valem para a rede municipal (escopo da tabela de metas)
    out.loc[out["rede"] != "Municipal", META_COLS] = np.nan
    out["meta_ano"] = _meta_do_ano(out)
    out["meta_2030"] = out["meta_alfabetizacao_2030"]
    out["gap_meta_ano"] = out["taxa_alfabetizacao"] - out["meta_ano"]
    out["gap_meta_2030"] = out["taxa_alfabetizacao"] - out["meta_2030"]
    out["atingiu_meta_ano"] = (out["gap_meta_ano"] >= 0).where(out["meta_ano"].notna())
    cols = ["ano", "id_municipio", "sigla_uf", "rede", "origem", "taxa_alfabetizacao", "media_portugues",
            "meta_ano", "meta_2030", "gap_meta_ano", "gap_meta_2030", "atingiu_meta_ano"] + META_COLS
    return out[cols].drop_duplicates(["ano", "id_municipio", "rede"]).reset_index(drop=True)


def check_against_official_gold(root: str, indicador: pd.DataFrame) -> dict:
    """Confere a tabela derivada com ``gold/comparacao_metas_resultados`` oficial."""
    try:
        gold = read_table(root, "gold", "comparacao_metas_resultados")
    except (FileNotFoundError, OSError):
        return {"status": "gold oficial indisponível"}
    gold["id_municipio"] = _norm_id(gold["id_municipio"])
    gold["ano"] = gold["ano"].astype(int)
    m = indicador[indicador["rede"] == "Municipal"].merge(gold, on=["id_municipio", "ano"], how="inner")
    return {
        "linhas_comparadas": int(len(m)),
        "max_dif_taxa": float((m["taxa_alfabetizacao"] - m["resultado_realizado"]).abs().max()) if len(m) else None,
        "max_dif_meta": float((m["meta_ano"] - m["meta_definida"]).abs().max()) if len(m) else None,
    }


def build_metas_vs_resultados_uf(root: str) -> pd.DataFrame:
    res = read_table(root, "silver", "resultado_uf", ["sigla_uf", "ano", "rede", "taxa_alfabetizacao", "media_portugues"])
    res["rede"] = _decode_rede(res["rede"])
    res["ano"] = res["ano"].astype(int)
    res = res[res["rede"] == "Pública"].dropna(subset=["taxa_alfabetizacao"])
    res = res.drop_duplicates(["sigla_uf", "ano"])[["ano", "sigla_uf", "taxa_alfabetizacao", "media_portugues"]]
    metas = read_table(root, "silver", "metas_uf")
    metas["sigla_uf"] = metas["sigla_uf"].astype(str).str.upper().str.strip()
    wide = _pivot_metas(metas, ["sigla_uf"])
    out = res.merge(wide, on="sigla_uf", how="left")
    out["meta_ano"] = _meta_do_ano(out)
    out["gap_meta_ano"] = out["taxa_alfabetizacao"] - out["meta_ano"]
    out["gap_meta_2030"] = out["taxa_alfabetizacao"] - out["meta_alfabetizacao_2030"]
    return out


def build_evolucao_temporal(root: str) -> pd.DataFrame:
    g = read_table(root, "gold", "evolucao_temporal_indicador")
    g = g.rename(columns={"percentual_alfabetizado_medio": "taxa_alfabetizacao", "sigla_uf": "unidade"})
    g["nivel"] = g["nivel"].replace({"BRASIL": "Brasil"})
    g["rede"] = "Pública"
    g["ano"] = g["ano"].astype(int)
    return g[["ano", "rede", "taxa_alfabetizacao", "nivel", "unidade"]].sort_values(["nivel", "unidade", "ano"])


def load_alunos_oficial(root: str, ano: int, columns: list[str] | None = None) -> pd.DataFrame:
    need = None
    if columns is not None:
        need = sorted(set(columns) | {"id_municipio", "rede"} - {"sigla_uf"})
    df = read_table(root, "silver", "alunos", need, filter_expr=ds.field("ano") == ano)
    if df.empty:
        raise FileNotFoundError(f"silver/alunos sem dados para ano={ano}")
    if len(df) < MIN_ALUNOS_ANO_COMPLETO:
        warnings.warn(
            f"silver/alunos ({ano}) tem {len(df):,} linhas — parece a AMOSTRA de 2% da Fase 2. "
            "Reextraia a tabela completa (remova QUERY_OVERRIDES['alunos']) para o modelo de alunos.",
            stacklevel=2,
        )
    df["id_municipio"] = _norm_id(df["id_municipio"])
    df["rede"] = _decode_rede(df["rede"])
    df["sigla_uf"] = df["id_municipio"].str[:2].astype(int).map(config.UF_IBGE)
    for c in ("presenca", "preenchimento_caderno", "alfabetizado"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int8")
    for c in ("proficiencia", "peso_aluno"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "presenca" in df and "proficiencia" in df:
        df.loc[df["presenca"] == 0, "proficiencia"] = np.nan
    if "id_aluno" in df:
        df = df.drop_duplicates(subset=["id_aluno"])
    df["ano"] = ano
    if columns is not None:
        df = df[[c for c in columns if c in df.columns] + ["ano"]]
    return df


def build_perfil_alunos_municipio(root: str, anos=(2023, 2024)) -> pd.DataFrame:
    """Mesma regra do perfil da Fase 2 local: agregados ponderados por peso_aluno."""
    frames = []
    for ano in anos:
        try:
            a = load_alunos_oficial(root, ano, ["id_municipio", "sigla_uf", "rede", "presenca",
                                                "alfabetizado", "proficiencia", "peso_aluno"])
        except FileNotFoundError:
            continue
        a["w"] = a["peso_aluno"].fillna(1.0)
        pres = a[a["presenca"] == 1].copy()
        pres["alf_w"] = pres["alfabetizado"] * pres["w"]
        pres["prof_w"] = pres["proficiencia"] * pres["w"]
        pres["corte_w"] = (pres["proficiencia"] >= config.PONTO_CORTE).astype(float) * pres["w"]
        keys = ["ano", "id_municipio", "sigla_uf", "rede"]
        g = pres.groupby(keys).agg(alunos_avaliados=("w", "size"), soma_pesos=("w", "sum"),
                                   alf_w=("alf_w", "sum"), prof_w=("prof_w", "sum"), corte_w=("corte_w", "sum")).reset_index()
        g["pct_alfabetizados"] = (100 * g["alf_w"] / g["soma_pesos"]).round(2)
        g["proficiencia_media"] = (g["prof_w"] / g["soma_pesos"]).round(2)
        g["pct_acima_corte_743"] = (100 * g["corte_w"] / g["soma_pesos"]).round(2)
        tot = a.groupby(keys).size().rename("alunos_matriculados").reset_index()
        g = g.merge(tot, on=keys, how="left")
        g["taxa_participacao"] = (100 * g["alunos_avaliados"] / g["alunos_matriculados"]).round(2)
        frames.append(g.drop(columns=["alf_w", "prof_w", "corte_w"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


BUILDERS = {
    "indicador_municipio": build_indicador_municipio,
    "metas_vs_resultados_uf": build_metas_vs_resultados_uf,
    "evolucao_temporal": build_evolucao_temporal,
    "perfil_alunos_municipio": build_perfil_alunos_municipio,
}


def materialize(root: str | None = None, force: bool = False) -> dict:
    """Gera ``data/analytic/*.parquet`` a partir do lake oficial e devolve um resumo."""
    root = str(root or config.DATALAKE)
    config.ANALYTIC_DIR.mkdir(parents=True, exist_ok=True)
    resumo = {}
    for name, fn in BUILDERS.items():
        dest = config.ANALYTIC_DIR / f"{name}.parquet"
        if dest.exists() and not force:
            continue
        df = fn(root)
        df.to_parquet(dest, index=False)
        resumo[name] = len(df)
    if "indicador_municipio" in resumo:
        resumo["validacao_gold_oficial"] = check_against_official_gold(
            root, pd.read_parquet(config.ANALYTIC_DIR / "indicador_municipio.parquet"))
    return resumo


if __name__ == "__main__":
    import json
    print(json.dumps(materialize(force=True), indent=2, ensure_ascii=False))
