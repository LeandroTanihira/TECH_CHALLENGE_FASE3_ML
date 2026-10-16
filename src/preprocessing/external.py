"""Enriquecimento com fontes externas (IBGE).

Fontes (dados públicos do IBGE redistribuídos em repositórios abertos):
* ``municipios_geo.csv``  — kelvins/municipios-brasileiros: código IBGE, lat/long, capital;
* ``brazil-population-1992-2019.csv`` — rdsilva/BrazilPopulation: estimativas
  populacionais do IBGE por município (1992–2019);
* ``vw_pib_percapita.csv`` — rdsilva/BrazilPopulation: PIB per capita municipal
  (IBGE, referência Censo 2010).

Gera ``data/external/municipios_enriquecidos.csv`` — uma linha por município.
"""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

from src import config

SOURCES = {
    "municipios_geo.csv": "https://raw.githubusercontent.com/kelvins/municipios-brasileiros/main/csv/municipios.csv",
    "brazil-population-1992-2019.csv": "https://raw.githubusercontent.com/rdsilva/BrazilPopulation/master/brazil-population-1992-2019.csv",
    "vw_pib_percapita.csv": "https://raw.githubusercontent.com/rdsilva/BrazilPopulation/master/vw_pib_percapita.csv",
}

OUTPUT = config.EXTERNAL_DIR / "municipios_enriquecidos.csv"


def download_sources(force: bool = False) -> None:
    """Baixa as fontes externas para ``data/raw`` (se ainda não existirem)."""
    import urllib.request

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fname, url in SOURCES.items():
        dest = config.RAW_DIR / fname
        if dest.exists() and not force:
            continue
        print(f"Baixando {fname} ...")
        urllib.request.urlretrieve(url, dest)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.upper().replace("-", " ").replace("'", " ").split())


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def build_external_features(force: bool = False) -> pd.DataFrame:
    if OUTPUT.exists() and not force:
        return pd.read_csv(OUTPUT, dtype={"id_municipio": str})
    download_sources()

    geo = pd.read_csv(config.RAW_DIR / "municipios_geo.csv")
    geo = geo.rename(columns={"codigo_ibge": "id_municipio", "nome": "nome_municipio"})
    geo["id_municipio"] = geo["id_municipio"].astype(str).str.zfill(7)
    geo["sigla_uf"] = geo["codigo_uf"].map(config.UF_IBGE)
    geo["regiao"] = geo["sigla_uf"].map(config.UF_REGIAO)
    geo["capital"] = geo["capital"].astype(int)

    # Distância até a capital do estado (proxy de acesso a serviços/infraestrutura)
    caps = geo[geo["capital"] == 1][["sigla_uf", "latitude", "longitude"]].rename(
        columns={"latitude": "lat_cap", "longitude": "lon_cap"}
    )
    geo = geo.merge(caps, on="sigla_uf", how="left")
    geo["dist_capital_km"] = _haversine_km(geo["latitude"], geo["longitude"], geo["lat_cap"], geo["lon_cap"]).round(1)

    # População (código IBGE de 6 dígitos na fonte)
    pop = pd.read_csv(config.RAW_DIR / "brazil-population-1992-2019.csv", sep=";")
    pop["cod6"] = pop["code"].astype(str).str.zfill(6)
    pop_w = pop.pivot_table(index="cod6", columns="year", values="population", aggfunc="first")
    pop_feat = pd.DataFrame({
        "populacao_2019": pop_w.get(2019),
        "cresc_pop_2012_2019_pct": 100 * (pop_w.get(2019) / pop_w.get(2012) - 1),
    }).reset_index()
    geo["cod6"] = geo["id_municipio"].str[:6]
    geo = geo.merge(pop_feat, on="cod6", how="left")

    # PIB per capita (chave: nome normalizado + UF)
    pib = pd.read_csv(
        config.RAW_DIR / "vw_pib_percapita.csv",
        usecols=["UF", "nome", "PIB_percapita"], encoding="latin1",
    )
    pib["UF"] = pib["UF"].str.encode("latin1", "ignore").str.decode("latin1")
    pib["sigla_uf"] = pib["UF"].map(lambda u: config.UF_NOME.get(u, config.UF_NOME.get(_norm(u))))
    norm_map = {_norm(k): v for k, v in config.UF_NOME.items()}
    pib["sigla_uf"] = pib["sigla_uf"].fillna(pib["UF"].map(lambda u: norm_map.get(_norm(u))))
    pib["chave"] = pib["nome"].map(_norm) + "|" + pib["sigla_uf"].astype(str)
    pib = pib.drop_duplicates("chave")[["chave", "PIB_percapita"]].rename(columns={"PIB_percapita": "pib_per_capita_2010"})
    geo["chave"] = geo["nome_municipio"].map(_norm) + "|" + geo["sigla_uf"]
    geo = geo.merge(pib, on="chave", how="left")

    out = geo[[
        "id_municipio", "nome_municipio", "sigla_uf", "regiao", "latitude", "longitude",
        "capital", "dist_capital_km", "populacao_2019", "cresc_pop_2012_2019_pct", "pib_per_capita_2010",
    ]].copy()
    out.to_csv(OUTPUT, index=False)
    return out


if __name__ == "__main__":
    df = build_external_features(force=True)
    print(df.shape)
    print(df.isna().mean().round(4))
