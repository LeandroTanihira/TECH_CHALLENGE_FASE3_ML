"""Leitura do data lake da Fase 2 (layout oficial AWS ou local) — ver ``src/config.py``."""
from __future__ import annotations

import pandas as pd

from src import config


def _read(path) -> pd.DataFrame:
    """Lê um dataset Parquet (arquivo único ou diretório particionado)."""
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.read_parquet(path)


def load_gold(name: str) -> pd.DataFrame:
    """Carrega uma tabela da camada analítica.

    Tabelas: indicador_municipio, perfil_alunos_municipio,
    metas_vs_resultados_uf, evolucao_temporal.

    * layout ``oficial``: derivada do lake da Fase 2 (S3/cópia local) pelo
      ``fase2_adapter`` e materializada em ``data/analytic/`` (cache);
    * layout ``local``: lida diretamente de ``<datalake>/gold/<name>``.
    """
    if config.FASE2_LAYOUT == "oficial":
        from src.preprocessing import fase2_adapter

        path = config.ANALYTIC_DIR / f"{name}.parquet"
        if not path.exists():
            fase2_adapter.materialize(str(config.DATALAKE))
        df = pd.read_parquet(path)
    else:
        df = _read(config.GOLD_DIR / name)
    if "id_municipio" in df:
        df["id_municipio"] = df["id_municipio"].astype(str).str.zfill(7)
    if "ano" in df:
        df["ano"] = df["ano"].astype(int)
    return df


def load_alunos(ano: int, columns: list[str] | None = None) -> pd.DataFrame:
    """Microdados de alunos (Silver) de um ano — ~2 milhões de linhas/ano."""
    if config.FASE2_LAYOUT == "oficial":
        from src.preprocessing import fase2_adapter

        return fase2_adapter.load_alunos_oficial(str(config.DATALAKE), ano, columns)

    base = config.SILVER_DIR / "alunos"
    candidates = [base / f"ano={ano}", base / f"alunos_{ano}.parquet"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Microdados de {ano} não encontrados em {base}")
    df = _read(path) if path.is_dir() else pd.read_parquet(path, columns=columns)
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]
    df["ano"] = ano
    if "id_municipio" in df:
        df["id_municipio"] = df["id_municipio"].astype(str).str.zfill(7)
    return df
