# Registro de decisões analíticas (ADR simplificado)

Cada decisão traz contexto, alternativas consideradas e consequência. Ordem cronológica do projeto.

| # | Decisão | Alternativas | Motivo | Evidência |
|---|---|---|---|---|
| 01 | Consumir o data lake da Fase 2 ([repositório](https://github.com/postechfiap018-ctrl/TECH_CHALLENGE_FASE2_PIPELINE)) via **adaptador** que gera uma camada analítica | Reescrever a Fase 3 só sobre as 3 tabelas Gold oficiais | A Gold oficial é agregada (sem proficiência, participação ou histórico por rede); a Silver oficial tem tudo o que é necessário. O adaptador é validado contra `gold/comparacao_metas_resultados` | `tests/test_fase2_adapter.py`; diferença máxima = 0 |
| 02 | Usar a tabela `alunos` **completa** (export da Base dos Dados sem amostragem, processado pela arquitetura medalhão da Fase 2 executada localmente) em vez da Silver AWS com `TABLESAMPLE 2%` | Usar a amostra de 2% | Com 2% (~40 mil alunos/ano), o contexto por escola fica inviável e o treino perde poder. A equivalência entre os dois caminhos é testada | README §3.3; `tests/test_fase2_adapter.py` |
| 03 | Unidade de modelagem = **aluno presente** | Incluir ausentes | Ausentes têm `alfabetizado=0` sem terem feito a prova → alvo falso | EDA §2 |
| 04 | **Contexto T-1 → alvo T** (2023 → 2024) | Usar taxas do mesmo ano | Taxas do mesmo ano contêm o resultado do próprio aluno (vazamento) e não existem no momento da decisão | EDA §8 |
| 05 | Remover `proficiencia`, `preenchimento_caderno`, `peso_aluno`, `presenca` | — | Pós-prova; `proficiencia ≥ 743` define o alvo (100% de concordância) | EDA §8 |
| 06 | Não usar `caderno` | Usar como categórica | Atribuição aleatória, sem sinal | EDA §8 |
| 07 | Split e CV **agrupados por município** | Split aleatório por aluno | Alunos do mesmo município compartilham contexto → métrica inflada | asserts em `train.split_by_group` |
| 08 | Pré-processamento **dentro do Pipeline** | Pré-processar a base inteira antes do split | Evita vazamento de estatísticas de imputação/escala | `src/modeling/pipelines.py` |
| 09 | Imputação pela mediana **com indicador de ausência** | Descartar linhas | Ausências são estruturais (SP/DF/AC/RR sem microdados 2023; escolas novas) | notebook 02 §1 |
| 10 | Remover `mun_meta_2030` | Manter | Constante (80%) para todos os municípios | EDA |
| 11 | Enriquecer com IBGE (população, PIB per capita, geografia) | Censo Escolar / IDHM | Acesso automatizado reprodutível; demais fontes ficam como evolução | `src/preprocessing/external.py` |
| 12 | Métrica principal **ROC-AUC** + acurácia balanceada, recall de "não alfabetizado" e Brier | Acurácia | Classes 60/40; o uso é ordenar risco e somar probabilidades (calibração importa) | notebook 02 |
| 13 | Limiar escolhido na **validação** (máx. F1-macro) | 0,5 fixo / escolher no teste | Equilibra as classes sem contaminar o teste | `metrics.best_threshold` |
| 14 | **LightGBM regularizado** como modelo final de alunos | RF, logística, HistGB | Melhor AUC após tuning, gap 3× menor que o padrão, SHAP exato | notebook 02 §4–6 |
| 15 | **Regressão logística** para risco municipal | Random Forest (AUC maior na CV estratificada) | RF cai de 0,787 para 0,621 na CV por UF; logística mantém 0,707 e é interpretável | notebook 04 §1 |
| 16 | Excluir do ranking municípios sem meta 2025 ou sem resultado 2024 | Imputar a meta | Alvo indefinido; ranquear seria inventar risco | `main.step_strategic` |
| 17 | K-Means com **k = 4** | k pela maior silhueta (k=2) | Silhueta varia pouco entre 4 e 7; 4 perfis são acionáveis para gestão | notebook 04 §3 |
| 18 | Não usar 2025 (streaming simulado da Fase 2) como alvo | Usar | Dados sintéticos | EDA §1 |
| 19 | Comunicar o modelo de alunos como **risco contextual**, não diagnóstico individual | — | AUC ≈ 0,66 no aluno; agregação municipal é precisa (r = 0,81) | notebook 03 §4 |
| 20 | Versionar uma **amostra do data lake** (`data/amostra/`) com saídas isoladas | Exigir o data lake da Fase 2 para qualquer execução | Quem corrige consegue rodar pipeline e notebooks sem AWS, sem sobrescrever os resultados oficiais | README §3.4 |
