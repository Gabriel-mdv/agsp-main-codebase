"""
utils/visualization.py
───────────────────────
All plotting utilities for AGSP-VLM (MobileVLM V2-1.7B).

Produces all figures used in the paper and appendices:

  Paper figures
  ─────────────
  plot_ais_distribution()     Figure 1: AIS histogram + top-20 heads bar chart
  plot_ais_heatmap()          Figure 2: AIS heatmap across 24 layers × 16 heads
  plot_gini_comparison()      Figure 3: Cross-scale Gini bar chart
  plot_epoch1_damage()        Figure 4: Epoch-1 loss across pruning ratios
  plot_perlayer_pattern()     Figure 5: Per-layer head removal at 40%
  plot_pareto_curve()         Figure 6: Final loss Pareto curve
  plot_convergence()          Figure 7: LoRA convergence at 40%
  plot_pope_results()         Figure 8: POPE accuracy / F1 / yes-ratio

  Appendix figures
  ────────────────
  plot_calibration_ablation() Appendix A: Spearman rho vs calibration size
  plot_attention_heatmap()    Attention cross-modal heatmap (Phase 1 diagnostic)
  plot_safety_radar()         nuScenes safety radar chart (H3)

All figures are saved as PNG at 150 DPI. Matplotlib backend is set to Agg
(non-interactive) for compatibility with headless PSC Bridges-2 nodes.

MobileVLM V2 architecture constants used in plots:
  n_layers = 24
  n_heads  = 16
  n_visual_tokens = 144  (LDPv2 projector)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # headless backend — required on PSC Bridges-2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Shared style constants
# ─────────────────────────────────────────────────────────────────────────────

STYLE = {
    "agsp_color":     "#2196F3",   # blue — AGSP throughout
    "mbp_color":      "#F44336",   # red  — MBP throughout
    "rand_color":     "#9E9E9E",   # grey — random baseline
    "baseline_color": "#4CAF50",   # green — unpruned baseline
    "highlight":      "#FF9800",   # orange — critical heads / annotations
    "fig_dpi":        150,
    "fig_width":      12,
    "font_size":      11,
    "title_size":     13,
}

# Critical layers in MobileVLM V2-1.7B (from Phase 1 results)
CRITICAL_LAYERS = [4, 18, 20, 22]


def _save(fig, path: str) -> None:
    """Save figure and close to free memory."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=STYLE["fig_dpi"], bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: AIS distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_ais_distribution(
    ais: np.ndarray,
    top_k_heads: List[Tuple[int, int, float]],
    save_path: str,
    gini: Optional[float] = None,
) -> None:
    """
    Figure 1: AIS distribution histogram (left) and top-20 heads bar chart (right).

    The right-skewed distribution with a small number of high-AIS heads
    confirms that grounding signal is concentrated in a few critical heads.
    """
    flat = ais.flatten()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(STYLE["fig_width"], 5))

    title = "AIS Distribution — MobileVLM V2-1.7B (384 heads)"
    if gini is not None:
        title += f"  |  Gini = {gini:.3f}"
    fig.suptitle(title, fontsize=STYLE["title_size"], fontweight="bold")

    # Left: histogram
    ax1.hist(flat, bins=60, color=STYLE["agsp_color"], alpha=0.8, edgecolor="white")
    ax1.set_yscale("log")
    ax1.set_xlabel("AIS Score", fontsize=STYLE["font_size"])
    ax1.set_ylabel("Count (log scale)", fontsize=STYLE["font_size"])
    ax1.set_title("All 384 Heads", fontsize=12)
    ax1.axvline(
        np.median(flat), color="black", linestyle="--", linewidth=1.2,
        label=f"Median = {np.median(flat):.3f}",
    )
    ax1.axvline(
        float(np.mean(flat)), color=STYLE["highlight"], linestyle="-.", linewidth=1.2,
        label=f"Mean = {float(np.mean(flat)):.3f}",
    )
    ax1.legend(fontsize=9)

    # Right: top-k bar chart
    labels  = [f"L{l}H{h}" for l, h, _ in top_k_heads]
    values  = [v for _, _, v in top_k_heads]
    colours = [
        STYLE["highlight"] if i < 5 else STYLE["agsp_color"]
        for i in range(len(values))
    ]
    y_pos = range(len(labels))
    ax2.barh(list(y_pos), values, color=colours, edgecolor="white")
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("AIS Score", fontsize=STYLE["font_size"])
    ax2.set_title(f"Top-{len(top_k_heads)} Alignment-Critical Heads", fontsize=12)
    if len(values) >= 5:
        ax2.axvline(values[4], color=STYLE["highlight"], linestyle="--",
                    linewidth=1, alpha=0.6, label="Top-5 threshold")
    ax2.legend(fontsize=9)

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: AIS heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_ais_heatmap(
    ais: np.ndarray,
    top_k_heads: List[Tuple[int, int, float]],
    save_path: str,
    top_k_mark: int = 5,
) -> None:
    """
    Figure 2: AIS heatmap across 24 layers × 16 heads.

    Stars mark the top-k alignment-critical heads.
    High-AIS layers (4, 18, 20, 22) are annotated.
    """
    n_layers, n_heads = ais.shape
    fig, ax = plt.subplots(figsize=(10, 7))

    im = ax.imshow(ais, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="AIS Score", fraction=0.03, pad=0.04)

    ax.set_xlabel("Head Index (0–15)", fontsize=STYLE["font_size"])
    ax.set_ylabel("Layer Index (0–23)", fontsize=STYLE["font_size"])
    ax.set_title(
        f"AIS Heatmap — MobileVLM V2-1.7B\n"
        f"{n_layers} layers × {n_heads} heads = {n_layers * n_heads} total heads  "
        f"★ = Top-{top_k_mark} critical heads",
        fontsize=STYLE["title_size"],
    )
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(0, n_layers, 2))

    # Mark top-k heads with stars
    for rank, (layer, head, score) in enumerate(top_k_heads[:top_k_mark]):
        ax.plot(
            head, layer, marker="*", color="white", markersize=14,
            markeredgecolor="black", markeredgewidth=0.8,
        )
        ax.text(
            head + 0.3, layer, f"★{rank+1}", color="white",
            fontsize=7, fontweight="bold", va="center",
        )

    # Annotate critical layers
    for layer in CRITICAL_LAYERS:
        ax.axhline(layer, color="white", linewidth=0.5, alpha=0.4, linestyle="--")

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Cross-scale Gini comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_gini_comparison(
    gini_values: Dict[str, float],
    save_path: str,
) -> None:
    """
    Figure 3: Gini coefficient comparison across model sizes.

    gini_values: dict mapping model label to Gini, e.g.
      {"MobileLLaMA-1.4B\n(MobileVLM V2)": 0.232, "LLaVA-1.5-7B": 0.540}

    Higher Gini = more concentrated alignment distribution = specialised heads.
    """
    labels = list(gini_values.keys())
    values = list(gini_values.values())
    colours = [STYLE["agsp_color"] if i == 0 else STYLE["mbp_color"]
               for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colours, alpha=0.85, edgecolor="white", width=0.5)

    # Annotate bar values
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1,
               label="Gini = 0.50 threshold")
    ax.set_ylabel("Gini Coefficient (AIS Distribution)", fontsize=STYLE["font_size"])
    ax.set_title(
        "Cross-Scale Gini Comparison\n"
        "Preliminary observation — not a scaling law (two data points)",
        fontsize=STYLE["title_size"],
    )
    ax.set_ylim(0, max(values) * 1.3)
    ax.legend(fontsize=9)

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Epoch-1 loss (pre-recovery damage)
# ─────────────────────────────────────────────────────────────────────────────

def plot_epoch1_damage(
    ratios: List[float],
    agsp_epoch1: List[float],
    mbp_epoch1: List[float],
    save_path: str,
    rand_epoch1: Optional[List[float]] = None,
) -> None:
    """
    Figure 4: Epoch-1 loss across pruning ratios.

    This is the paper's primary criterion quality metric — lower Epoch-1 loss
    means less grounding damage from the pruning criterion before any recovery.
    AGSP should consistently show ~2× lower loss than MBP.
    """
    ratios_pct = [r * 100 for r in ratios]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ratios_pct, agsp_epoch1, color=STYLE["agsp_color"],
            marker="o", linewidth=2.5, markersize=8, label="AGSP (alignment-guided)")
    ax.plot(ratios_pct, mbp_epoch1, color=STYLE["mbp_color"],
            marker="s", linewidth=2.5, markersize=8, linestyle="--",
            label="MBP (magnitude-guided)")
    if rand_epoch1 is not None:
        ax.plot(ratios_pct, rand_epoch1, color=STYLE["rand_color"],
                marker="^", linewidth=1.5, markersize=7, linestyle=":",
                label="Random (lower bound)")

    # Annotate the 2× gap at each ratio
    for i, (r, a, m) in enumerate(zip(ratios_pct, agsp_epoch1, mbp_epoch1)):
        if m > 0:
            ratio_label = f"{m/a:.1f}×"
            ax.annotate(
                ratio_label,
                xy=(r, (a + m) / 2),
                fontsize=8, color="black", ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            )

    ax.set_xlabel("Pruning Ratio (%)", fontsize=STYLE["font_size"])
    ax.set_ylabel("Epoch-1 Loss (immediate pruning damage)", fontsize=STYLE["font_size"])
    ax.set_title(
        "Epoch-1 Loss: Immediate Grounding Damage Before Fine-Tuning Recovery\n"
        "Lower = less grounding damage from pruning criterion",
        fontsize=STYLE["title_size"],
    )
    ax.set_xticks(ratios_pct)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Per-layer pruning pattern
# ─────────────────────────────────────────────────────────────────────────────

def plot_perlayer_pattern(
    agsp_per_layer: List[int],
    mbp_per_layer: List[int],
    save_path: str,
    pruning_ratio: float = 0.40,
    max_cap: int = 8,
    wasserstein_dist: Optional[float] = None,
) -> None:
    """
    Figure 5: Per-layer head removal at a given pruning ratio.

    Shows the structural difference between AGSP (distributed across all
    layers) and MBP (front-loaded in early layers). Critical layers are
    highlighted. Wasserstein distance quantifies the distributional difference.
    """
    n_layers = len(agsp_per_layer)
    layers   = list(range(n_layers))
    width    = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(STYLE["fig_width"], 5),
                                    sharey=True)
    total = sum(agsp_per_layer)
    fig.suptitle(
        f"Per-Layer Head Removal Pattern: AGSP vs MBP at {int(pruning_ratio*100)}% Pruning\n"
        f"Same {total} heads removed — different selection strategy"
        + (f"  |  Wasserstein distance = {wasserstein_dist:.3f}" if wasserstein_dist else ""),
        fontsize=STYLE["title_size"], fontweight="bold",
    )

    # Colour bars: critical layers highlighted in red, others in blue/grey
    def _bar_colours(per_layer, base_colour):
        return [STYLE["mbp_color"] if l in CRITICAL_LAYERS else base_colour
                for l in range(len(per_layer))]

    # AGSP panel
    colours_agsp = _bar_colours(agsp_per_layer, STYLE["agsp_color"])
    ax1.bar(layers, agsp_per_layer, color=colours_agsp, alpha=0.85, edgecolor="white")
    ax1.axhline(max_cap, color="red", linestyle="--", linewidth=1.2,
                label=f"Max cap ({max_cap} heads)")
    ax1.set_xlabel("Layer Index", fontsize=STYLE["font_size"])
    ax1.set_ylabel("Heads Removed", fontsize=STYLE["font_size"])
    ax1.set_title(
        f"AGSP @ {int(pruning_ratio*100)}% — AIS-guided\n(all {n_layers} layers affected)",
        fontsize=11,
    )
    ax1.set_xticks(range(0, n_layers, 2))
    ax1.legend(fontsize=9)

    # Mark critical layers
    for cl in CRITICAL_LAYERS:
        ax1.annotate("★", xy=(cl, agsp_per_layer[cl] + 0.2),
                     ha="center", fontsize=10, color=STYLE["highlight"])

    # MBP panel
    colours_mbp = _bar_colours(mbp_per_layer, STYLE["mbp_color"])
    ax2.bar(layers, mbp_per_layer, color=colours_mbp, alpha=0.85, edgecolor="white")
    ax2.axhline(max_cap, color="red", linestyle="--", linewidth=1.2,
                label=f"Max cap ({max_cap} heads)")
    ax2.set_xlabel("Layer Index", fontsize=STYLE["font_size"])
    ax2.set_title(
        f"MBP @ {int(pruning_ratio*100)}% — Magnitude-guided\n(front-loaded)",
        fontsize=11,
    )
    ax2.set_xticks(range(0, n_layers, 2))
    ax2.legend(fontsize=9)

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Pareto curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_pareto_curve(
    ratios: List[float],
    agsp_losses: List[float],
    mbp_losses: List[float],
    metric_name: str,
    save_path: str,
    rand_losses: Optional[List[float]] = None,
) -> None:
    """
    Figure 6: Final epoch loss vs pruning ratio (Pareto curve).

    AGSP should maintain a flat curve while MBP degrades sharply at high
    compression. The shaded region shows AGSP's advantage.
    """
    ratios_pct = [r * 100 for r in ratios]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ratios_pct, agsp_losses, color=STYLE["agsp_color"],
            marker="o", linewidth=2.5, markersize=8, label="AGSP")
    ax.plot(ratios_pct, mbp_losses, color=STYLE["mbp_color"],
            marker="s", linewidth=2.5, markersize=8, linestyle="--", label="MBP")
    if rand_losses is not None:
        ax.plot(ratios_pct, rand_losses, color=STYLE["rand_color"],
                marker="^", linewidth=1.5, markersize=7, linestyle=":",
                label="Random")

    # Shade AGSP advantage region
    ax.fill_between(
        ratios_pct, agsp_losses, mbp_losses,
        where=[a <= m for a, m in zip(agsp_losses, mbp_losses)],
        alpha=0.12, color=STYLE["agsp_color"], label="AGSP advantage",
    )

    # Annotate Δ at each ratio
    for r, a, m in zip(ratios_pct, agsp_losses, mbp_losses):
        delta = m - a
        if abs(delta) > 0.001:
            ax.annotate(
                f"Δ={delta:+.3f}",
                xy=(r, (a + m) / 2),
                fontsize=7, ha="center", color="black",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8),
            )

    ax.set_xlabel("Head Pruning Ratio (%)", fontsize=STYLE["font_size"])
    ax.set_ylabel(metric_name, fontsize=STYLE["font_size"])
    ax.set_title(
        f"Pareto Curve: Compression vs {metric_name}\n"
        "MobileVLM V2-1.7B — 3-epoch LoRA recovery",
        fontsize=STYLE["title_size"],
    )
    ax.set_xticks(ratios_pct)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: LoRA convergence
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(
    agsp_losses: List[float],
    mbp_losses: List[float],
    save_path: str,
    pruning_ratio: float = 0.40,
    rand_losses: Optional[List[float]] = None,
) -> None:
    """
    Figure 7: LoRA fine-tuning convergence at a given pruning ratio.

    Shows how Epoch-1 loss (grounding damage) narrows over 3 epochs of
    LoRA recovery. The gap at Epoch 1 is the key diagnostic; the gap at
    Epoch 3 shows AGSP's advantage in final recovery quality.
    """
    epochs = list(range(1, len(agsp_losses) + 1))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, agsp_losses, color=STYLE["agsp_color"],
            marker="o", linewidth=2.5, markersize=9, label="AGSP")
    ax.plot(epochs, mbp_losses, color=STYLE["mbp_color"],
            marker="s", linewidth=2.5, markersize=9, linestyle="--", label="MBP")
    if rand_losses is not None:
        ax.plot(epochs, rand_losses, color=STYLE["rand_color"],
                marker="^", linewidth=1.5, markersize=7, linestyle=":",
                label="Random")

    # Annotate Epoch-1 values
    ax.annotate(
        f"E1: {agsp_losses[0]:.4f}",
        xy=(1, agsp_losses[0]), xytext=(1.15, agsp_losses[0] + 0.05),
        fontsize=9, color=STYLE["agsp_color"], fontweight="bold",
    )
    ax.annotate(
        f"E1: {mbp_losses[0]:.4f}",
        xy=(1, mbp_losses[0]), xytext=(1.15, mbp_losses[0] + 0.05),
        fontsize=9, color=STYLE["mbp_color"], fontweight="bold",
    )

    # Annotate 2× gap at Epoch 1
    if mbp_losses[0] > 0:
        gap = mbp_losses[0] / agsp_losses[0]
        ax.annotate(
            f"{gap:.1f}× gap",
            xy=(1, (agsp_losses[0] + mbp_losses[0]) / 2),
            xytext=(1.4, (agsp_losses[0] + mbp_losses[0]) / 2),
            fontsize=9, color="black",
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.2),
        )

    ax.set_xlabel("Epoch", fontsize=STYLE["font_size"])
    ax.set_ylabel("Mean Training Loss", fontsize=STYLE["font_size"])
    ax.set_title(
        f"LoRA Fine-Tuning Convergence at {int(pruning_ratio*100)}% Pruning\n"
        "COCO val2017 captions · 6,000 samples · batch=4 · LR=1e-5",
        fontsize=STYLE["title_size"],
    )
    ax.set_xticks(epochs)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: POPE results
# ─────────────────────────────────────────────────────────────────────────────

def plot_pope_results(
    results: Dict[str, Dict],
    save_path: str,
) -> None:
    """
    Figure 8: POPE adversarial evaluation results.

    Three subplots: accuracy, F1, yes-ratio (output calibration).
    results dict keys: 'baseline', 'agsp', 'mbp'
    Each value has keys: accuracy, f1, yes_ratio.
    """
    labels  = list(results.keys())
    colours = {
        "baseline": STYLE["baseline_color"],
        "agsp":     STYLE["agsp_color"],
        "mbp":      STYLE["mbp_color"],
    }
    bar_colours = [colours.get(k, STYLE["agsp_color"]) for k in labels]

    metrics = [
        ("accuracy",  "POPE Accuracy",         "Higher is better"),
        ("f1",        "POPE F1",                "Higher is better"),
        ("yes_ratio", "Yes-Ratio (calibration)","Closer to baseline is better"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(STYLE["fig_width"], 5))
    fig.suptitle(
        "POPE Adversarial Evaluation — MobileVLM V2-1.7B @ 40% Pruning\n"
        "After output recalibration (Stage 2 fine-tuning)",
        fontsize=STYLE["title_size"], fontweight="bold",
    )

    for ax, (key, title, note) in zip(axes, metrics):
        values = [results[l].get(key, 0) for l in labels]
        bars   = ax.bar(labels, values, color=bar_colours, alpha=0.85,
                        edgecolor="white", width=0.5)

        # Annotate bar values
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

        # Add baseline reference line
        baseline_val = results.get("baseline", {}).get(key, None)
        if baseline_val is not None:
            ax.axhline(baseline_val, color=STYLE["baseline_color"],
                       linestyle="--", linewidth=1.2, alpha=0.7,
                       label=f"Baseline ({baseline_val:.3f})")
            ax.legend(fontsize=8)

        ax.set_title(f"{title}\n{note}", fontsize=10)
        ax.set_ylim(0, max(values) * 1.25)
        ax.set_xticklabels(labels, fontsize=9)

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Appendix A: Calibration set size ablation
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration_ablation(
    sizes: List[int],
    rhos: List[float],
    save_path: str,
    threshold: float = 0.85,
) -> None:
    """
    Appendix A: Spearman rho vs calibration set size.

    Shows that AIS rankings are stable across calibration set sizes.
    All points should be above the threshold line (rho >= 0.85).
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes, rhos, color=STYLE["agsp_color"], marker="o",
            linewidth=2.5, markersize=9, label="Spearman ρ")

    # Annotate rho values
    for size, rho in zip(sizes, rhos):
        ax.annotate(
            f"{rho:.4f}",
            xy=(size, rho), xytext=(size, rho + 0.005),
            ha="center", fontsize=9, fontweight="bold",
        )

    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2,
               label=f"Threshold (ρ = {threshold})")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Calibration Set Size (number of COCO image-caption pairs)",
                  fontsize=STYLE["font_size"])
    ax.set_ylabel("Spearman ρ vs 500-pair ranking", fontsize=STYLE["font_size"])
    ax.set_title(
        "Appendix A: AIS Stability vs Calibration Set Size\n"
        "Ground truth: 500-pair AIS ranking",
        fontsize=STYLE["title_size"],
    )
    ax.set_xticks(sizes)
    ax.set_ylim(threshold - 0.05, 1.02)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Attention heatmap (Phase 1 diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention_heatmap(
    attn_weights: np.ndarray,
    caption: str,
    save_path: str,
) -> None:
    """
    Cross-modal attention heatmap for one sample (diagnostic only).

    Shows text-to-visual attention patterns. Used in Phase 1 to verify
    that the model is attending to visual tokens correctly.

    NOTE: This is for visualisation only and is NOT used to compute AIS.
    AIS uses gradient Frobenius norms, not raw attention weights.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(attn_weights, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_xlabel("Visual Token Index (LDPv2 — 144 tokens)", fontsize=STYLE["font_size"])
    ax.set_ylabel("Text Token Index", fontsize=STYLE["font_size"])
    caption_display = caption if len(caption) <= 60 else caption[:57] + "..."
    ax.set_title(
        f"Cross-Modal Attention Heatmap\n\"{caption_display}\"",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Attention weight")

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer AIS profile
# ─────────────────────────────────────────────────────────────────────────────

def plot_layer_ais_profile(
    ais: np.ndarray,
    save_path: str,
) -> None:
    """
    Per-layer mean and max AIS profile.

    Identifies which layers contain the most alignment-critical heads.
    Critical layers (4, 18, 20, 22) should have the highest mean and max AIS.
    """
    n_layers    = ais.shape[0]
    layer_means = ais.mean(axis=1)
    layer_maxes = ais.max(axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(n_layers), layer_means, color=STYLE["agsp_color"],
           alpha=0.7, label="Mean AIS per layer")
    ax.plot(range(n_layers), layer_maxes, color=STYLE["highlight"],
            marker="o", markersize=4, linewidth=1.5, label="Max AIS per layer")

    # Highlight critical layers
    for cl in CRITICAL_LAYERS:
        ax.axvspan(cl - 0.4, cl + 0.4, alpha=0.15, color=STYLE["highlight"])
        ax.text(cl, layer_maxes[cl] + 0.02, f"L{cl}",
                ha="center", fontsize=8, color=STYLE["highlight"], fontweight="bold")

    ax.set_xlabel("Layer Index", fontsize=STYLE["font_size"])
    ax.set_ylabel("AIS", fontsize=STYLE["font_size"])
    ax.set_title(
        "Per-Layer AIS Profile — MobileVLM V2-1.7B\n"
        "Highlighted layers contain the most alignment-critical heads",
        fontsize=STYLE["title_size"],
    )
    ax.set_xticks(range(0, n_layers, 2))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# nuScenes safety radar (H3)
# ─────────────────────────────────────────────────────────────────────────────

def plot_safety_radar(
    baseline_safety: Dict[str, float],
    agsp_safety: Dict[str, float],
    mbp_safety: Dict[str, float],
    save_path: str,
) -> None:
    """
    nuScenes safety radar chart (H3 hypothesis).

    Shows per-category detection accuracy for baseline, AGSP, and MBP.
    Categories: pedestrian, vehicle, stop sign, traffic light, cyclist.
    """
    categories = [k for k in baseline_safety if k != "average"]
    n = len(categories)
    if n < 3:
        logger.warning("Too few categories for radar chart (need >= 3).")
        return

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for data, label, colour in [
        (baseline_safety, "Baseline (unpruned)", STYLE["baseline_color"]),
        (agsp_safety,     "AGSP",                STYLE["agsp_color"]),
        (mbp_safety,      "MBP",                 STYLE["mbp_color"]),
    ]:
        values = [data.get(c, 0) for c in categories] + [data.get(categories[0], 0)]
        ax.plot(angles, values, color=colour, linewidth=2, label=label)
        ax.fill(angles, values, color=colour, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title(
        "H3: Safety-Critical Object Detection\nMobileVLM V2-1.7B @ 40% Pruning",
        fontsize=STYLE["title_size"], pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# AGSP vs MBP comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_agsp_vs_mbp_comparison(
    results: Dict[str, Dict],
    save_path: str,
) -> None:
    """
    Summary bar chart comparing baseline, AGSP, and MBP on key metrics.

    results dict: keys are model labels, values are dicts with
    keys: pope, vram, latency.
    """
    model_labels = [k for k in results if k in ["baseline", "agsp", "mbp"]]
    metrics = ["POPE Acc.", "VRAM (GB)", "Latency (ms)"]

    def _get(res, key):
        return res.get(key, 0)

    baseline_vals = [_get(results.get("baseline", {}), k)
                     for k in ["pope", "vram", "latency"]]
    agsp_vals     = [_get(results.get("agsp", {}), k)
                     for k in ["pope", "vram", "latency"]]
    mbp_vals      = [_get(results.get("mbp", {}), k)
                     for k in ["pope", "vram", "latency"]]

    x = np.arange(len(metrics))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, baseline_vals, w, label="Baseline (unpruned)",
           color=STYLE["baseline_color"], alpha=0.85)
    ax.bar(x,     agsp_vals,     w, label="AGSP (40%)",
           color=STYLE["agsp_color"],     alpha=0.85)
    ax.bar(x + w, mbp_vals,      w, label="MBP (40%)",
           color=STYLE["mbp_color"],      alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=STYLE["font_size"])
    ax.set_ylabel("Score / Value", fontsize=STYLE["font_size"])
    ax.set_title(
        "AGSP vs MBP vs Baseline — MobileVLM V2-1.7B",
        fontsize=STYLE["title_size"],
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    _save(fig, save_path)