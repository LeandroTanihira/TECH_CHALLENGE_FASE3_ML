# Amostra versionada do data lake

Permite executar a pipeline e os notebooks **sem acesso ao data lake da Fase 2**.

| Conteúdo | Origem |
|---|---|
| `gold/` | camada Gold completa (4 tabelas analíticas, ~1 MB) |
| `silver/alunos/alunos_2023.parquet`, `alunos_2024.parquet` | todos os alunos de **555 municípios** sorteados (10% por UF, semente 42) — 203.583 (2023) e 291.353 (2024) registros |

Uso: é selecionada automaticamente quando nenhum data lake é encontrado, ou com `FASE3_AMOSTRA=1`.
As saídas vão para `outputs_amostra/` (fora do Git) — os resultados oficiais em `reports/`, `models/` e `images/`
foram gerados com a base **completa** e não são sobrescritos.

Os números obtidos com a amostra diferem levemente dos oficiais (menos municípios no treino/teste), mas o
fluxo, as validações e as conclusões são os mesmos.
