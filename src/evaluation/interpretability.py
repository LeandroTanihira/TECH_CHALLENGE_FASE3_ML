"""Interpretabilidade: Feature Importance (permutação e ganho) e SHAP Values."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import RANDOM_STATE


def transformed_matrix(pipe, X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Aplica todas as etapas do pipeline exceto o modelo e devolve os nomes das colunas."""
    Xt = X
    for _, step in pipe.steps[:-1]:
        Xt = step.transform(Xt)
    names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    return pd.DataFrame(np.asarray(Xt, dtype=float), columns=names, index=X.index), names


def model_importance(pipe) -> pd.DataFrame:
    """Importância nativa do modelo (ganho, para boosting; impureza, para RF)."""
    model = pipe.named_steps["model"]
    names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    if hasattr(model, "booster_"):
        imp = model.booster_.feature_importance(importance_type="gain")
    elif hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    else:
        raise ValueError("modelo sem importância nativa")
    out = pd.DataFrame({"feature": names, "importancia": imp})
    out["importancia_pct"] = 100 * out["importancia"] / out["importancia"].sum()
    return out.sort_values("importancia", ascending=False).reset_index(drop=True)


def permutation_importance_df(pipe, X: pd.DataFrame, y, n_repeats=5, scoring="roc_auc") -> pd.DataFrame:
    """Queda de AUC ao embaralhar cada variável ORIGINAL (no conjunto de teste)."""
    r = permutation_importance(pipe, X, y, n_repeats=n_repeats, scoring=scoring,
                               random_state=RANDOM_STATE, n_jobs=1)
    return (pd.DataFrame({"feature": X.columns, "queda_auc_media": r.importances_mean,
                          "queda_auc_desvio": r.importances_std})
            .sort_values("queda_auc_media", ascending=False).reset_index(drop=True))


def shap_values(pipe, X: pd.DataFrame, max_rows=5000):
    """SHAP (TreeExplainer) sobre a matriz já pré-processada.

    Retorna (explanation, Xt). Valores em log-odds da classe "alfabetizado".
    """
    import shap

    Xs = X.sample(n=min(max_rows, len(X)), random_state=RANDOM_STATE)
    Xt, names = transformed_matrix(pipe, Xs)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    sv = explainer(Xt)
    if sv.values.ndim == 3:  # alguns modelos retornam (n, features, classes)
        sv = sv[:, :, 1]
    return sv, Xt


def shap_summary(sv) -> pd.DataFrame:
    vals = np.abs(sv.values).mean(axis=0)
    out = pd.DataFrame({"feature": sv.feature_names, "shap_medio_abs": vals})
    return out.sort_values("shap_medio_abs", ascending=False).reset_index(drop=True)
