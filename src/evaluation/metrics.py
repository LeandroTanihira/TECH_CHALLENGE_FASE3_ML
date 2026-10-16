"""Métricas de avaliação de classificação binária."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)


def classification_report_dict(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        # Classe de risco (não alfabetizado = 0)
        "recall_nao_alfabetizado": tn / (tn + fp) if (tn + fp) else np.nan,
        "precision_nao_alfabetizado": tn / (tn + fn) if (tn + fn) else np.nan,
        "f1_macro": f1_score(y_true, pred, average="macro"),
        "brier": brier_score_loss(y_true, proba),
        "log_loss": log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y_true)),
    }


def best_threshold(y_true, proba, criterion: str = "f1_macro") -> float:
    """Escolhe o limiar na VALIDAÇÃO (nunca no teste)."""
    grid = np.linspace(0.30, 0.75, 46)
    if criterion == "f1_macro":
        scores = [f1_score(y_true, (proba >= t).astype(int), average="macro") for t in grid]
    elif criterion == "balanced_accuracy":
        scores = [balanced_accuracy_score(y_true, (proba >= t).astype(int)) for t in grid]
    else:  # f1 da classe positiva
        p, r, t = precision_recall_curve(y_true, proba)
        f = 2 * p * r / np.clip(p + r, 1e-9, None)
        return float(t[np.nanargmax(f[:-1])])
    return float(grid[int(np.argmax(scores))])


def metrics_by_group(df: pd.DataFrame, group: str, y="y", p="proba", threshold=0.5) -> pd.DataFrame:
    """Desempenho por subgrupo (região, rede...) — checagem de equidade do modelo."""
    rows = []
    for g, d in df.groupby(group, observed=True):
        if d[y].nunique() < 2:
            continue
        m = classification_report_dict(d[y], d[p].values, threshold)
        rows.append({group: g, "n": len(d), "taxa_real": d[y].mean(), "taxa_prevista": d[p].mean(),
                     "roc_auc": m["roc_auc"], "balanced_accuracy": m["balanced_accuracy"],
                     "recall_nao_alfabetizado": m["recall_nao_alfabetizado"]})
    return pd.DataFrame(rows)
