"""Testa o adaptador do lake oficial da Fase 2 contra um lake emulado.

Uso:  BASE_FASE2=<pasta com os CSVs da Base dos Dados> pytest tests/ -q
"""
import os
from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing import fase2_adapter as fa
from tests.fase2_oficial_emulator import build

BASE = Path(os.environ.get("BASE_FASE2", Path(__file__).resolve().parents[2] / "Base" / "Fase 2"))
pytestmark = pytest.mark.skipif(not (BASE / "municipio.csv").exists(), reason="CSVs da Base dos Dados ausentes")


@pytest.fixture(scope="module")
def lake(tmp_path_factory):
    return str(build(BASE, tmp_path_factory.mktemp("lake_oficial")))


def test_layout_detectado(lake):
    assert fa.is_official_layout(lake)


def test_indicador_bate_com_gold_oficial(lake):
    ind = fa.build_indicador_municipio(lake)
    chk = fa.check_against_official_gold(lake, ind)
    assert chk["linhas_comparadas"] > 4000
    assert chk["max_dif_taxa"] < 1e-9 and chk["max_dif_meta"] < 1e-9
    assert set(ind["rede"]) >= {"Municipal", "Estadual", "Pública"}
    assert ind["id_municipio"].str.len().eq(7).all()


def test_metas_uf_e_evolucao(lake):
    uf = fa.build_metas_vs_resultados_uf(lake)
    assert uf["sigla_uf"].nunique() >= 24 and uf["meta_alfabetizacao_2030"].notna().any()
    ev = fa.build_evolucao_temporal(lake)
    assert {"Brasil", "UF"} <= set(ev["nivel"])
