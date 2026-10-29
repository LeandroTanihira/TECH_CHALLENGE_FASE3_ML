# Roteiro — Vídeo executivo (até 5 minutos)

**Formato:** reunião executiva com gestores públicos (Secretaria de Educação / MEC). Tom: problema → evidência → decisão.
Telas sugeridas entre colchetes (imagens em `images/`).

---

### 0:00 – 0:40 · O problema
> "Em 2024, só **59 de cada 100 crianças** da rede pública terminaram o 2º ano alfabetizadas. A meta do país é **80 até 2030**.
> Apenas **10 dos 24 estados** com dados atingiram a meta de 2024. A pergunta que trazemos hoje não é *onde estamos* — é
> **onde vamos falhar se nada mudar, e por quê**."

[`eda_alvo_tendencia.png` → `eda_uf_2024.png`]

### 0:40 – 1:20 · O que fizemos
> "Usamos a plataforma de dados construída na fase anterior — resultados, metas, território e os microdados **completos** de alunos
> (2,1 milhões em 2024; a extração na nuvem usava amostra de 2%, então reprocessamos a tabela inteira pela mesma arquitetura) —
> e acrescentamos dados do IBGE. Treinamos modelos que usam apenas o que se sabe **antes** da prova: o histórico do ano anterior,
> as metas e o contexto do município. Validamos em municípios que o modelo **nunca viu**."

[diagrama da pipeline no README]

### 1:20 – 2:30 · Principais insights
1. > "**Gestão pesa mais que riqueza.** O histórico da rede municipal explica ~70% do que o modelo aprende; PIB per capita quase nada.
   > O Ceará, com renda baixa, é o 1º do país (85%)." [`interp_temas.png`]
2. > "**Engajamento é alavanca barata:** redes com alta participação na avaliação alfabetizam mais — de 48% a 71% entre os extremos." [`eda_hipoteses_quintis.png`]
3. > "**Choques importam:** o Sul caiu em 2024, puxado pelo Rio Grande do Sul após as enchentes." [`eda_regioes.png`]
4. > "**Perfis, não só regiões:** 35% dos municípios do Nordeste estão no grupo *Referência*; há municípios vulneráveis no Sul." [`estr_clusters.png`]

### 2:30 – 3:40 · O valor estratégico
> "O modelo individual não serve para rotular crianças — e não é esse o uso. Somando as probabilidades, ele prevê a taxa de cada
> município com **erro médio de 8 pontos**, **29% melhor** que supor que tudo se repete. Com isso, entregamos:"
* > "Um **ranking de risco para a meta de 2025**: 1.000 municípios em risco crítico e 945 em risco alto." [`estr_mapa_risco.png`]
* > "Uma **análise de trajetória até 2030**: 38% dos municípios estão estagnados ou em queda." [`estr_trajetoria_2030.png`]
* > "**Explicações** para cada priorização (SHAP) — o gestor vê o porquê, não só a nota." [`interp_shap_waterfall.png`]

### 3:40 – 4:40 · Como isso apoia a política pública
1. Focalizar assistência técnica nos municípios críticos com mais crianças.
2. Plano de recomposição de aprendizagem no RS e revisão pactuada das metas.
3. Meta de participação ≥ 90% na avaliação.
4. Cooperação entre municípios *Referência* e *Vulneráveis* do mesmo porte (modelo PAIC).
5. Re-treinar a cada ciclo do Saeb e acompanhar no painel.

### 4:40 – 5:00 · Fechamento
> "Com dados públicos e uma pipeline reprodutível, é possível **antecipar** onde a alfabetização vai falhar e **agir antes**.
> Os limites — poucos ciclos e falta de dados individuais — estão documentados, e o próximo passo é integrar o Censo Escolar."

---
**Checklist de gravação:** câmera aberta do apresentador · slides com as figuras de `images/` · cronômetro ≤ 5:00 · link do repositório na tela final.
