"""Configurações centrais do projeto (caminhos, constantes e sementes).

A Fase 3 consome o data lake (Silver/Gold) produzido pela pipeline da Fase 2:
https://github.com/postechfiap018-ctrl/TECH_CHALLENGE_FASE2_PIPELINE

Ordem de resolução do data lake:

1. ``FASE2_DATALAKE`` — caminho local ou URI ``s3://bucket`` (prefixos silver/ e gold/);
2. ``DATALAKE_BUCKET`` — mesma variável do repositório da Fase 2 → ``s3://<bucket>``;
3. ``data/fase2_oficial/`` — cópia local do bucket (``aws s3 sync``, ver README);
4. ``../Desafio - Fase 2/datalake`` — execução local da pipeline da Fase 2;
5. ``data/`` deste repositório (cópia local opcional: ``data/gold`` e ``data/silver``);
6. ``data/amostra/`` — **amostra versionada** (Gold completa + alunos de ~10% dos municípios),
   usada automaticamente quando nenhum data lake está disponível (ex.: quem clona o
   repositório para corrigir). Force com ``FASE3_AMOSTRA=1``. Nesse modo as saídas vão para
   ``outputs_amostra/`` para não sobrescrever os resultados oficiais versionados.

Dois *layouts* são suportados e detectados automaticamente:

* ``oficial`` — schema da pipeline AWS entregue (``gold/comparacao_metas_resultados`` …);
  convertido para a camada analítica por ``src/preprocessing/fase2_adapter.py``;
* ``local``   — schema analítico já pronto (``gold/indicador_municipio`` …).
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FASE2_REPO_URL = "https://github.com/postechfiap018-ctrl/TECH_CHALLENGE_FASE2_PIPELINE"


AMOSTRA_DIR = PROJECT_ROOT / "data" / "amostra"


def _resolve_datalake():
    if os.environ.get("FASE3_AMOSTRA") == "1":
        return AMOSTRA_DIR
    env = os.environ.get("FASE2_DATALAKE")
    if env:
        return env if env.startswith("s3://") else Path(env)
    bucket = os.environ.get("DATALAKE_BUCKET")
    if bucket:
        return f"s3://{bucket}"
    for cand in (PROJECT_ROOT / "data" / "fase2_oficial",
                 PROJECT_ROOT.parent / "Desafio - Fase 2" / "datalake",
                 PROJECT_ROOT / "data"):
        if (cand / "gold").exists():
            return cand
    return AMOSTRA_DIR


def _detect_layout(root) -> str:
    forced = os.environ.get("FASE2_LAYOUT")
    if forced in ("oficial", "local"):
        return forced
    if str(root).startswith("s3://") or (Path(root) / "gold" / "comparacao_metas_resultados").exists():
        return "oficial"
    return "local"


DATALAKE = _resolve_datalake()
FASE2_LAYOUT = _detect_layout(DATALAKE)
GOLD_DIR = Path(DATALAKE) / "gold" if not str(DATALAKE).startswith("s3://") else None
SILVER_DIR = Path(DATALAKE) / "silver" if not str(DATALAKE).startswith("s3://") else None

MODO_AMOSTRA = Path(str(DATALAKE)) == AMOSTRA_DIR

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # fontes externas baixadas (IBGE via GitHub)
EXTERNAL_DIR = DATA_DIR / "external"  # fontes externas tratadas
ANALYTIC_DIR = DATA_DIR / "analytic"  # camada analítica derivada do lake oficial
if MODO_AMOSTRA:
    # Saídas separadas: a amostra nunca sobrescreve os resultados oficiais versionados
    _OUT = PROJECT_ROOT / "outputs_amostra"
    PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, IMAGES_DIR = (_OUT / d for d in ("processed", "models", "reports", "images"))
    print(f"[config] MODO AMOSTRA: data lake = {AMOSTRA_DIR} · saídas em {_OUT}")
else:
    PROCESSED_DIR = DATA_DIR / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    IMAGES_DIR = PROJECT_ROOT / "images"

for _d in (EXTERNAL_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, IMAGES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Reprodutibilidade
RANDOM_STATE = 42

# Desenho temporal: contexto (features) de um ano prevê o resultado do ano seguinte
CONTEXT_YEAR = 2023
TARGET_YEAR = 2024

# Ponto de corte Saeb para alfabetização (INEP, 2023)
PONTO_CORTE = 743.0

# UF -> região geográfica
UF_REGIAO = {
    "AC": "Norte", "AP": "Norte", "AM": "Norte", "PA": "Norte", "RO": "Norte", "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste", "PB": "Nordeste",
    "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste", "MS": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}

UF_IBGE = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL",
    28: "SE", 29: "BA", 31: "MG", 32: "ES", 33: "RJ", 35: "SP", 41: "PR",
    42: "SC", 43: "RS", 50: "MS", 51: "MT", 52: "GO", 53: "DF",
}

UF_NOME = {
    "RONDÔNIA": "RO", "ACRE": "AC", "AMAZONAS": "AM", "RORAIMA": "RR", "PARÁ": "PA", "AMAPÁ": "AP",
    "TOCANTINS": "TO", "MARANHÃO": "MA", "PIAUÍ": "PI", "CEARÁ": "CE", "RIO GRANDE DO NORTE": "RN",
    "PARAÍBA": "PB", "PERNAMBUCO": "PE", "ALAGOAS": "AL", "SERGIPE": "SE", "BAHIA": "BA",
    "MINAS GERAIS": "MG", "ESPIRITO SANTO": "ES", "ESPÍRITO SANTO": "ES", "RIO DE JANEIRO": "RJ",
    "SÃO PAULO": "SP", "PARANÁ": "PR", "SANTA CATARINA": "SC", "RIO GRANDE DO SUL": "RS",
    "MATO GROSSO DO SUL": "MS", "MATO GROSSO": "MT", "GOIÁS": "GO", "DISTRITO FEDERAL": "DF",
}
