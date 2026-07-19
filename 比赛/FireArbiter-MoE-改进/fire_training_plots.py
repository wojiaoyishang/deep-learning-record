# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _series(history: List[Dict], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, np.nan)) for row in history], dtype=np.float64)


def _has_finite(history: List[Dict], keys: Sequence[str]) -> bool:
    return any(np.isfinite(_series(history, key)).any() for key in keys)


def _pretty_label(key: str) -> str:
    text = str(key)
    if text.startswith("train_"):
        text = "Train " + text[len("train_"):]
    elif text.startswith("val_"):
        text = "Val " + text[len("val_"):]
    return text.replace("_", " ")


def _plot_lines(
    ax,
    history: List[Dict],
    keys: Sequence[str],
    title: str,
    ylabel: str = "Value",
    legend_columns: int = 2,
) -> None:
    x = _series(history, "epoch")
    plotted = False
    for key in keys:
        values = _series(history, key)
        if np.isfinite(values).any():
            ax.plot(x, values, marker="o", markersize=2.8, linewidth=1.35, label=_pretty_label(key))
            plotted = True
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(fontsize=7.5, ncol=max(1, legend_columns))
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)


def _save_dashboard(
    history: List[Dict],
    panels: Sequence[Tuple[str, Sequence[str], str]],
    title: str,
    path: Path,
    columns: int = 2,
) -> None:
    if not history:
        return
    visible_panels = [panel for panel in panels if _has_finite(history, panel[1])]
    if not visible_panels:
        return
    rows = int(math.ceil(len(visible_panels) / max(columns, 1)))
    fig, axes = plt.subplots(rows, columns, figsize=(7.2 * columns, 4.3 * rows), squeeze=False)
    for ax, (panel_title, keys, ylabel) in zip(axes.flat, visible_panels):
        _plot_lines(ax, history, keys, panel_title, ylabel)
    for ax in axes.flat[len(visible_panels):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_training_curves(output_dir: Path, history: List[Dict], expert_names: Sequence[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 各损失使用独立子图，避免量级不同的曲线挤在同一坐标轴中。
    _save_dashboard(
        history,
        [
            ("Total loss", ["train_loss", "val_loss"], "Loss"),
            ("Classification loss", ["train_loss_cls", "val_loss_cls"], "Loss"),
            ("Branch loss", ["train_loss_branch", "val_loss_branch"], "Loss"),
            ("Attention map loss", ["train_loss_map", "val_loss_map"], "Loss"),
            ("Expert loss", ["train_loss_expert", "val_loss_expert"], "Loss"),
            ("Calibration loss", ["train_loss_calibration", "val_loss_calibration"], "Loss"),
            ("Selected expert loss", ["train_loss_expert_selected", "val_loss_expert_selected"], "Loss"),
            ("Router KL / confidence loss", [
                "train_loss_expert_posterior_kl", "val_loss_expert_posterior_kl",
                "train_loss_expert_confidence", "val_loss_expert_confidence",
            ], "Loss"),
        ],
        "FireArbiter-MoE-Lite loss dashboard",
        output_dir / "loss_curves_latest.png",
        columns=2,
    )

    _save_dashboard(
        history,
        [
            ("Accuracy", ["train_acc", "val_acc", "train_balanced_acc", "val_balanced_acc"], "Score"),
            ("Precision / recall / F1", [
                "train_precision", "val_precision", "train_recall", "val_recall", "train_f1", "val_f1",
            ], "Score"),
            ("AUC", ["train_auc", "val_auc"], "AUC"),
            ("Calibration metrics", ["train_ece", "val_ece", "train_brier", "val_brier"], "Value"),
            ("Best threshold", ["train_best_threshold", "val_best_threshold"], "Threshold"),
            ("Branch weights", ["train_global_weight", "val_global_weight", "train_local_weight", "val_local_weight"], "Weight"),
        ],
        "Classification and calibration metrics",
        output_dir / "metric_curves_latest.png",
        columns=2,
    )

    _save_dashboard(
        history,
        [
            ("Semantic evidence", [
                "train_semantic_positive", "val_semantic_positive",
                "train_semantic_negative", "val_semantic_negative",
                "train_semantic_suspicious", "val_semantic_suspicious",
            ], "Score"),
            ("Semantic / expert confidence", [
                "train_semantic_confidence", "val_semantic_confidence",
                "train_expert_confidence_mean", "val_expert_confidence_mean",
            ], "Confidence"),
            ("Router behavior", [
                "train_router_change_rate", "val_router_change_rate",
                "train_posterior_improvement", "val_posterior_improvement",
            ], "Value"),
            ("Router entropy", [
                "train_router_prior_entropy", "val_router_prior_entropy",
                "train_router_posterior_entropy", "val_router_posterior_entropy",
            ], "Entropy"),
            ("Attention inside ratio", [
                "train_global_inside_ratio", "val_global_inside_ratio",
                "train_local_inside_ratio", "val_local_inside_ratio",
                "train_semantic_inside_ratio", "val_semantic_inside_ratio",
                "train_mix_inside_ratio", "val_mix_inside_ratio",
            ], "Ratio"),
            ("Attention far leakage", [
                "train_global_far_leak", "val_global_far_leak",
                "train_local_far_leak", "val_local_far_leak",
                "train_semantic_far_leak", "val_semantic_far_leak",
                "train_mix_far_leak", "val_mix_far_leak",
            ], "Leakage"),
        ],
        "Semantic, router and attention diagnostics",
        output_dir / "diagnostic_curves_latest.png",
        columns=2,
    )

    # 专家历史曲线按指标拆分。每个子图只呈现一种含义，避免权重与准确率混画。
    expert_panels = []
    for metric, title, ylabel in (
        ("prior_weight", "Prior routing weights", "Mean weight"),
        ("posterior_weight", "Posterior routing weights", "Mean weight"),
        ("accuracy", "Expert accuracy", "Accuracy"),
        ("confidence", "Expert confidence", "Confidence"),
    ):
        keys = []
        for index, _ in enumerate(expert_names):
            keys.extend([f"train_expert_{index}_{metric}", f"val_expert_{index}_{metric}"])
        expert_panels.append((title, keys, ylabel))
    _save_dashboard(
        history,
        expert_panels,
        "Expert routing and performance history",
        output_dir / "expert_curves_latest.png",
        columns=2,
    )


def _normalize_shares(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total <= 0.0:
        return np.full_like(array, 1.0 / max(len(array), 1))
    return array / total


def _pie_autopct(percent: float) -> str:
    return f"{percent:.1f}%" if percent >= 2.0 else ""


def save_expert_epoch(
    output_dir: Path,
    epoch: int,
    stage: str,
    expert_names: Sequence[str],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
) -> None:
    expert_count = len(expert_names)
    x = np.arange(expert_count)
    train_prior = [train_metrics.get(f"expert_{i}_prior_weight", 0.0) for i in range(expert_count)]
    train_post = [train_metrics.get(f"expert_{i}_posterior_weight", 0.0) for i in range(expert_count)]
    val_prior = [val_metrics.get(f"expert_{i}_prior_weight", 0.0) for i in range(expert_count)]
    val_post = [val_metrics.get(f"expert_{i}_posterior_weight", 0.0) for i in range(expert_count)]
    val_accuracy = [val_metrics.get(f"expert_{i}_accuracy", 0.0) for i in range(expert_count)]
    val_confidence = [val_metrics.get(f"expert_{i}_confidence", 0.0) for i in range(expert_count)]

    fig, axes = plt.subplots(2, 2, figsize=(max(15, expert_count * 1.8), 12))

    width = 0.20
    axes[0, 0].bar(x - 1.5 * width, train_prior, width, label="train prior")
    axes[0, 0].bar(x - 0.5 * width, train_post, width, label="train posterior")
    axes[0, 0].bar(x + 0.5 * width, val_prior, width, label="val prior")
    axes[0, 0].bar(x + 1.5 * width, val_post, width, label="val posterior")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(expert_names, rotation=30, ha="right")
    axes[0, 0].set_ylim(0.0, max(1.0, max(train_prior + train_post + val_prior + val_post + [0.1]) * 1.15))
    axes[0, 0].set_title("Routing weight comparison")
    axes[0, 0].set_ylabel("Mean weight")
    axes[0, 0].grid(True, axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=2)

    train_shares = _normalize_shares(train_post)
    axes[0, 1].pie(train_shares, labels=None, autopct=_pie_autopct, startangle=90)
    axes[0, 1].set_title("Train posterior routing share")
    axes[0, 1].legend(expert_names, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)

    val_shares = _normalize_shares(val_post)
    axes[1, 0].pie(val_shares, labels=None, autopct=_pie_autopct, startangle=90)
    axes[1, 0].set_title("Validation posterior routing share")
    axes[1, 0].legend(expert_names, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)

    performance_width = 0.36
    axes[1, 1].bar(x - performance_width / 2, val_accuracy, performance_width, label="val accuracy")
    axes[1, 1].bar(x + performance_width / 2, val_confidence, performance_width, label="val confidence")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(expert_names, rotation=30, ha="right")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_title("Validation expert performance")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].grid(True, axis="y", alpha=0.25)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(f"Expert routing - epoch {epoch} - {stage}", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    expert_dir = output_dir / "expert_plots"
    expert_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(expert_dir / f"epoch_{epoch:03d}_{stage}.png", dpi=170)
    plt.close(fig)
