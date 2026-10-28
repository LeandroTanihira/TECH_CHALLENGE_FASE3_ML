# Checklist de entrega — Tech Challenge Fase 3

Mapa entre o que o enunciado pede e onde está no repositório.

## Entregáveis finais

| Pedido no enunciado | Onde está | Status |
|---|---|---|
| Repositório Git completo | este repositório (histórico com branches e PRs) | ✅ |
| Código-fonte organizado | `src/preprocessing`, `src/modeling`, `src/evaluation`, `src/visualization` | ✅ |
| Notebooks ou scripts | `notebooks/01` → `04` (executados, com saídas) · `python -m src.main` | ✅ |
| README detalhado | [`README.md`](../README.md) — todas as seções pedidas | ✅ |
| Documentação técnica | [`decisoes_analiticas.md`](decisoes_analiticas.md), docstrings, [`CONTRIBUTING.md`](../CONTRIBUTING.md) | ✅ |
| Vídeo executivo (até 5 min) | roteiro em [`roteiro_video.md`](roteiro_video.md) · **link:** _preencher após a gravação_ | ⏳ |
| Pipeline reproduzível | `src/main.py`, sementes fixas, `requirements.txt`, CI (`.github/workflows/ci.yml`) | ✅ |
| Análises e visualizações | `images/` (27 figuras), `reports/` (métricas, rankings, clusters, painel) | ✅ |

## Requisitos da pipeline de ML

| Requisito | Implementação |
|---|---|
| Imputação de valores faltantes (numéricas) | `SimpleImputer(strategy="median", add_indicator=True)` — `src/preprocessing/transformers.py` |
| Transformação de numéricas e categóricas | `StandardScaler`, `OneHotEncoder`, `OrdinalEncoder` em `ColumnTransformer` |
| Tratamento de data leakage | contexto T-1 → alvo T, remoção de variáveis pós-prova, split agrupado por município — README §5 |
| Pré-processamento integrado ao modelo | `Pipeline([FeatureEngineer, ColumnTransformer, modelo])` — `src/modeling/pipelines.py` |
| Treinamento e validação | `GroupKFold`, `RandomizedSearchCV`, limiar na validação, teste usado uma vez — `src/modeling/train.py` |
| Replicabilidade e generalização | sementes fixas, estabilidade entre sementes (σ AUC = 0,0005), CV por UF no modelo de risco |
| Interpretabilidade (Feature Importance e SHAP) | `src/evaluation/interpretability.py`, notebook 03 |

## Estrutura mínima do repositório

| Exigido | Presente |
|---|---|
| `data/` | ✅ (`amostra/` versionada para rodar sem o data lake, `external/`; dados volumosos gerados pela pipeline) |
| `notebooks/` | ✅ |
| `src/preprocessing`, `src/modeling`, `src/evaluation`, `src/visualization` | ✅ |
| `reports/`, `images/` | ✅ |
| `requirements.txt`, `README.md`, `.gitignore` | ✅ |

## Boas práticas de versionamento

| Exigido | Evidência |
|---|---|
| Histórico de commits | commits pequenos com convenção `tipo(escopo): descrição` |
| Uso de branches | `feature/*`, `docs/*`, `chore/*` (preservadas no remoto) |
| Pull requests | um PR por branch, com template e CI |
| Documentação das decisões analíticas | [`decisoes_analiticas.md`](decisoes_analiticas.md) |
| Organização colaborativa | `CONTRIBUTING.md`, templates de PR e issue, CI obrigatório |

## Antes de entregar
- [ ] Preencher nomes e RMs dos integrantes no README
- [ ] Gravar o vídeo e colar o link aqui e no README (seção "Vídeo executivo")
- [ ] Conferir se o CI está verde na `main`
- [ ] Confirmar que o repositório está público (ou com acesso dado aos professores)
