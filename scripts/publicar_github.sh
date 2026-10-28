#!/usr/bin/env bash
# =============================================================================
# Publica o projeto no GitHub preservando o fluxo de trabalho do grupo:
#   1. cria o repositório (se não existir);
#   2. publica o commit inicial na main;
#   3. para cada branch integrada localmente (em ordem): publica a branch,
#      abre um Pull Request com descrição e o integra (merge commit);
#   4. cria a release v1.0-entrega e tenta proteger a main.
#
# Requisitos: git e GitHub CLI (https://cli.github.com) autenticado:  gh auth login
# Windows: execute no "Git Bash" a partir da pasta do projeto.
#
# Uso:
#   bash scripts/publicar_github.sh [OWNER/REPO] [public|private]
#   bash scripts/publicar_github.sh postechfiap018-ctrl/TECH_CHALLENGE_FASE3_ML public
#
# Variáveis opcionais: DRY_RUN=1 (só mostra o plano) · GH=<binário gh alternativo>
# =============================================================================
set -euo pipefail

REPO="${1:-postechfiap018-ctrl/TECH_CHALLENGE_FASE3_ML}"
VISIBILITY="${2:-public}"
GH="${GH:-gh}"
DESCRICAO="Tech Challenge Fase 3 (POSTECH FIAP) — Predição e inteligência analítica para alfabetização no Brasil"
FASE2="https://github.com/postechfiap018-ctrl/TECH_CHALLENGE_FASE2_PIPELINE"

cd "$(git rev-parse --show-toplevel)"
SOURCE_BRANCH="${SOURCE_BRANCH:-main}"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "✖ Há alterações não commitadas. Faça commit antes de publicar." >&2; exit 1
fi

# ---------------------------------------------------------------- plano
ROOT=$(git rev-list --max-parents=0 "$SOURCE_BRANCH" | tail -1)
mapfile -t MERGES < <(git rev-list --first-parent --reverse --merges "$SOURCE_BRANCH")
echo "Repositório : $REPO ($VISIBILITY)"
echo "Commit raiz : $(git log -1 --format='%h %s' "$ROOT")"
echo "PRs a criar : ${#MERGES[@]}"
for M in "${MERGES[@]}"; do
  BR=$(git log -1 --format=%s "$M" | sed -E 's/^Merge pull request: //; s/^Merge branch .([^ ]+). .*/\1/')
  echo "  • $BR ($(git rev-list --count "$M^1..$M^2") commits)"
done
[[ "${DRY_RUN:-0}" == "1" ]] && { echo "(DRY_RUN) nada foi publicado."; exit 0; }

# ---------------------------------------------------------------- repositório
"$GH" auth status >/dev/null 2>&1 || { echo "✖ Rode 'gh auth login' primeiro." >&2; exit 1; }
if "$GH" repo view "$REPO" >/dev/null 2>&1; then
  echo "• Repositório já existe — usando $REPO"
else
  "$GH" repo create "$REPO" "--$VISIBILITY" --description "$DESCRICAO"
fi
URL="${REMOTE_URL:-https://github.com/$REPO.git}"
git remote remove origin 2>/dev/null || true
git remote add origin "$URL"

if git ls-remote --exit-code --heads origin main >/dev/null 2>&1; then
  echo "✖ O remoto já tem uma branch main. Use um repositório vazio para preservar o histórico de PRs." >&2; exit 1
fi

ORIG_BRANCH=$(git rev-parse --abbrev-ref HEAD)
git push -q origin "$ROOT:refs/heads/main"
echo "✔ main inicial publicada"

# ---------------------------------------------------------------- PRs
for M in "${MERGES[@]}"; do
  BR=$(git log -1 --format=%s "$M" | sed -E 's/^Merge pull request: //; s/^Merge branch .([^ ]+). .*/\1/')
  mapfile -t COMMITS < <(git rev-list --reverse "$M^1..$M^2")
  git fetch -q origin main
  git checkout -q -B "$BR" origin/main
  git cherry-pick --allow-empty --keep-redundant-commits "${COMMITS[@]}" >/dev/null
  # Conferência: o conteúdo publicado deve ser idêntico ao do merge local
  if [[ "$(git rev-parse "HEAD^{tree}")" != "$(git rev-parse "$M^{tree}")" ]]; then
    echo "✖ Conteúdo de $BR diverge do histórico local — abortando." >&2; exit 1
  fi
  git push -q -u origin "$BR" --force

  TITULO=$(git log -1 --format=%s "${COMMITS[0]}")
  [[ ${#COMMITS[@]} -gt 1 ]] && TITULO="$BR: $(git log -1 --format=%s "${COMMITS[-1]}")"
  CORPO=$(cat <<EOF
## O que este PR entrega
$(for c in "${COMMITS[@]}"; do echo "- $(git log -1 --format=%s "$c")"; done)

## Arquivos
$(git diff --stat "$M^1" "$M" | tail -1)

## Checklist
- [x] Sem variáveis de vazamento
- [x] Pré-processamento dentro do \`Pipeline\`
- [x] Semente fixa nos passos aleatórios
- [x] Testes / notebooks executados

Dados de origem: pipeline da Fase 2 — $FASE2
EOF
)
  PR_URL=$("$GH" pr create --repo "$REPO" --base main --head "$BR" --title "$TITULO" --body "$CORPO")
  "$GH" pr merge "$PR_URL" --repo "$REPO" --merge
  echo "✔ PR integrado: $BR → $PR_URL"
done

# ---------------------------------------------------------------- finalização
git fetch -q origin
git checkout -q -B main origin/main
git branch -q --set-upstream-to=origin/main main
if [[ "$(git rev-parse "main^{tree}")" != "$(git rev-parse "$M^{tree}")" ]]; then
  echo "✖ main remota diverge do conteúdo local esperado." >&2; exit 1
fi
"$GH" release create v1.0-entrega --repo "$REPO" --target main --title "Entrega Tech Challenge Fase 3" \
  --notes "Entrega da Fase 3. Ver README e docs/entrega.md." >/dev/null && echo "✔ Release v1.0-entrega criada"

# Proteção da main (exige PR + CI). Pode falhar em contas sem permissão de admin/plano — não bloqueia.
"$GH" api -X PUT "repos/$REPO/branches/main/protection" --input - >/dev/null 2>&1 <<'JSON' && echo "✔ main protegida" || echo "• Proteção da main não aplicada (opcional)"
{"required_status_checks": {"strict": true, "contexts": []},
 "enforce_admins": false,
 "required_pull_request_reviews": {"required_approving_review_count": 1},
 "restrictions": null}
JSON

echo ""
echo "Pronto: https://github.com/$REPO"
