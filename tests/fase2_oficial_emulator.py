"""Emula, em pandas, o data lake gerado pela pipeline oficial da Fase 2
(Glue Silver/Gold — https://github.com/postechfiap018-ctrl/TECH_CHALLENGE_FASE2_PIPELINE)
a partir dos CSVs da Base dos Dados. Serve para testar o ``fase2_adapter``
sem credenciais AWS. Reproduz as regras de ``glue_silver_job.py`` e
``glue_gold_job.py``: chaves normalizadas, rede como código, metas despivotadas
(ano_meta, meta_valor) com a versão mais recente de cada meta, partições Hive.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

META_COLS = [f"meta_alfabetizacao_{a}" for a in range(2024, 2031)]
REDE_TXT_TO_COD = {"Estadual": 2, "Municipal": 3, "Privada": 4, "Federal": 1}


def _write(df: pd.DataFrame, root: Path, layer: str, name: str, partition: str | None = None):
    path = root / layer / name
    path.mkdir(parents=True, exist_ok=True)
    if partition:
        for val, part in df.groupby(partition):
            p = path / f"{partition}={val}"
            p.mkdir(exist_ok=True)
            part.drop(columns=[partition]).to_parquet(p / "part-0.parquet", index=False)
    else:
        df.to_parquet(path / "part-0.parquet", index=False)


def _unpivot(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    df = df.rename(columns={"ano": "ano_base"})
    long = df.melt(id_vars=keys + ["ano_base"], value_vars=[c for c in META_COLS if c in df],
                   var_name="ano_meta", value_name="meta_valor")
    long["ano_meta"] = long["ano_meta"].str[-4:].astype(int)
    return long.sort_values("ano_base").drop_duplicates(keys + ["ano_meta"], keep="last")


def build(base_dir: Path, out: Path, alunos: pd.DataFrame | None = None) -> Path:
    mun = pd.read_csv(base_dir / "municipio.csv")
    mun["id_municipio"] = mun["id_municipio"].astype(str).str.zfill(7)
    mun = mun.drop_duplicates(["id_municipio", "ano", "rede", "serie"])
    mun["sigla_uf"] = None  # a Silver oficial traz sigla_uf da dimensão territorial
    _write(mun, out, "silver", "resultado_municipio", "ano")

    uf = pd.read_csv(base_dir / "uf.csv").drop_duplicates(["sigla_uf", "ano", "rede", "serie"])
    _write(uf, out, "silver", "resultado_uf", "ano")

    mm = pd.read_csv(base_dir / "meta_alfabetizacao_municipio.csv")
    mm["id_municipio"] = mm["id_municipio"].astype(str).str.zfill(7)
    mm_long = _unpivot(mm, ["id_municipio", "rede"])
    _write(mm_long, out, "silver", "metas_municipio")
    mu = pd.read_csv(base_dir / "meta_alfabetizacao_uf.csv")
    _write(_unpivot(mu, ["sigla_uf", "rede"]), out, "silver", "metas_uf")

    if alunos is not None:
        a = alunos.copy()
        a["rede"] = a["rede"].map(REDE_TXT_TO_COD)
        _write(a.drop(columns=["sigla_uf"], errors="ignore"), out, "silver", "alunos", "ano")

    # GOLD oficial
    mun3 = mun[mun["rede"] == 3]
    comp = mun3.merge(mm_long, left_on=["id_municipio", "ano"], right_on=["id_municipio", "ano_meta"])
    comp = comp.rename(columns={"taxa_alfabetizacao": "resultado_realizado", "meta_valor": "meta_definida"})
    comp["gap_percentual"] = comp["resultado_realizado"] - comp["meta_definida"]
    comp = comp[["id_municipio", "sigla_uf", "ano", "resultado_realizado", "meta_definida", "gap_percentual"]]
    _write(comp, out, "gold", "comparacao_metas_resultados", "ano")

    uf5 = uf[uf["rede"] == 5]
    ev_uf = uf5.groupby(["sigla_uf", "ano"], as_index=False)["taxa_alfabetizacao"].mean()
    ev_uf["nivel"] = "UF"
    ev_br = uf5.groupby("ano", as_index=False)["taxa_alfabetizacao"].mean()
    ev_br["sigla_uf"], ev_br["nivel"] = "BR", "BRASIL"
    ev = pd.concat([ev_uf, ev_br]).rename(columns={"taxa_alfabetizacao": "percentual_alfabetizado_medio"})
    _write(ev, out, "gold", "evolucao_temporal_indicador", "nivel")
    return out
