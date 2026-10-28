# Como contribuir

## Fluxo de trabalho
1. Abra (ou pegue) uma issue descrevendo a tarefa.
2. Crie uma branch a partir da `main` atualizada:
   - `feature/<tema>` — nova funcionalidade (ex.: `feature/modelagem`)
   - `fix/<tema>` — correção
   - `docs/<tema>` — documentação
   - `chore/<tema>` — estrutura, configuração, CI
3. Faça commits pequenos e descritivos (convenção abaixo).
4. Abra um **Pull Request** para a `main` usando o template; peça revisão de pelo menos um integrante.
5. O PR só é integrado com o **CI verde** e a revisão aprovada (merge commit, sem *squash*, para preservar o histórico).

## Convenção de commits
`<tipo>(<escopo opcional>): <descrição no imperativo>` — tipos: `feat`, `fix`, `docs`, `test`, `chore`, `refactor`.
Ex.: `feat(train): split agrupado por município e GroupKFold`.

## Regras do projeto
- Nunca use variáveis de vazamento (ver `docs/decisoes_analiticas.md`, decisões 04–05).
- Todo pré-processamento fica dentro do `Pipeline` do scikit-learn.
- Decisões analíticas novas entram em `docs/decisoes_analiticas.md`.
- Dados volumosos e caches não são versionados (`.gitignore`); tudo deve ser reprodutível com `python -m src.main`.
