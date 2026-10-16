# Fontes externas (IBGE)

Baixadas automaticamente por `python -m src.preprocessing.external` (não versionadas):

| Arquivo | Conteúdo | Origem |
|---|---|---|
| `municipios_geo.csv` | código IBGE, latitude/longitude, capital | https://github.com/kelvins/municipios-brasileiros |
| `brazil-population-1992-2019.csv` | estimativas populacionais municipais (IBGE) | https://github.com/rdsilva/BrazilPopulation |
| `vw_pib_percapita.csv` | PIB per capita municipal (IBGE, ref. Censo 2010) | https://github.com/rdsilva/BrazilPopulation |

O resultado tratado fica em `data/external/municipios_enriquecidos.csv`.
