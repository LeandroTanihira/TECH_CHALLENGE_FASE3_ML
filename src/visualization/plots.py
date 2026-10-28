"""Estilo e gráficos padronizados (matplotlib).

Paleta categórica validada para daltonismo (ordem fixa — nunca reciclar),
rampa sequencial azul e par divergente azul ↔ vermelho com meio cinza.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src import config

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS = {"Baixo": "#0ca30c", "Moderado": "#fab219", "Alto": "#ec835a", "Crítico": "#d03b3b"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ)
DIV_CMAP = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a32b2b"])
REGIAO_COR = {"Norte": CAT[0], "Nordeste": CAT[1], "Centro-Oeste": CAT[2], "Sudeste": CAT[6], "Sul": CAT[4]}


def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
        "axes.prop_cycle": plt.cycler(color=CAT), "figure.dpi": 110, "savefig.dpi": 150,
        "savefig.bbox": "tight", "lines.linewidth": 2,
    })


def save(fig, name: str):
    path = config.IMAGES_DIR / f"{name}.png"
    fig.savefig(path)
    return path


def barh(series: pd.Series, title: str, xlabel: str = "", color=CAT[0], fmt="{:.1f}", ax=None, top=None):
    """Barras horizontais ordenadas com rótulo direto nas extremidades."""
    s = series.sort_values()
    if top:
        s = s.tail(top)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(s) + 1)))
    ax.barh(s.index.astype(str), s.values, color=color, height=0.7)
    ax.grid(axis="y", visible=False)
    span = (s.max() - min(0, s.min())) or 1
    for i, v in enumerate(s.values):
        ax.text(v + span * 0.01, i, fmt.format(v), va="center", fontsize=8, color=INK2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    return ax


def roc_pr_curves(results: dict, y_true):
    """Curvas ROC e Precisão-Revocação para vários modelos (mesmo conjunto)."""
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, (name, proba) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_true, proba)
        axes[0].plot(fpr, tpr, color=CAT[i], label=f"{name} (AUC {auc(fpr, tpr):.3f})")
        p, r, _ = precision_recall_curve(y_true, proba)
        axes[1].plot(r, p, color=CAT[i], label=name)
    axes[0].plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    axes[0].set(title="Curva ROC", xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos")
    axes[1].axhline(np.mean(y_true), ls="--", color=MUTED, lw=1)
    axes[1].set(title="Precisão × Revocação", xlabel="Revocação", ylabel="Precisão")
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    return fig


def calibration_plot(y_true, proba, n_bins=10, ax=None, label="modelo"):
    from sklearn.calibration import calibration_curve

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    frac, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1, label="calibração perfeita")
    ax.plot(mean_pred, frac, marker="o", ms=6, color=CAT[0], label=label)
    ax.set(title="Calibração das probabilidades", xlabel="Probabilidade prevista", ylabel="Frequência observada")
    ax.legend(loc="upper left")
    return ax


def confusion_plot(tn, fp, fn, tp, ax=None, title="Matriz de confusão (teste)"):
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4))
    m = np.array([[tn, fp], [fn, tp]])
    ax.imshow(m, cmap=SEQ_CMAP)
    ax.grid(False)
    labels = ["Não alfabetizado", "Alfabetizado"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set(xlabel="Previsto", ylabel="Real", title=title)
    tot = m.sum()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]:,}\n({m[i, j] / tot:.1%})".replace(",", "."), ha="center", va="center",
                    color="white" if m[i, j] > m.max() * 0.55 else INK, fontsize=9)
    return ax
