"""Matplotlib chart generation for Telegram dashboard visuals.

Generates PnL equity curves, win/loss pie charts, and sport-breakdown
bar charts. Returns bytes (PNG) suitable for Telegram send_photo().
"""
from __future__ import annotations

import io
import logging
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

log = logging.getLogger(__name__)

# Consistent dark theme for premium look
DARK_BG = "#1a1a2e"
CARD_BG = "#16213e"
ACCENT = "#0f3460"
GREEN = "#2ecc71"
RED = "#e74c3c"
GOLD = "#f39c12"
TEXT_COLOR = "#ecf0f1"
GRID_COLOR = "#2c3e50"


def _apply_dark_theme(fig, ax):
    """Apply consistent dark theme to figure and axes."""
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_COLOR, which="both")
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.3, linestyle="--")


def generate_pnl_chart(
    equity_curve: List[float],
    initial_bankroll: float = 1000.0,
    title: str = "Bankroll-Entwicklung",
) -> bytes:
    """Generate a PnL equity curve chart. Returns PNG bytes."""
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_theme(fig, ax)

    x = list(range(len(equity_curve)))
    y = equity_curve

    # Color the line green if above initial, red if below
    ax.fill_between(x, initial_bankroll, y,
                    where=[v >= initial_bankroll for v in y],
                    alpha=0.3, color=GREEN, interpolate=True)
    ax.fill_between(x, initial_bankroll, y,
                    where=[v < initial_bankroll for v in y],
                    alpha=0.3, color=RED, interpolate=True)
    ax.plot(x, y, color=GOLD, linewidth=2, zorder=5)
    ax.axhline(y=initial_bankroll, color=TEXT_COLOR, linestyle="--", alpha=0.5, linewidth=1)

    # Annotations
    if y:
        final = y[-1]
        pnl = final - initial_bankroll
        pnl_pct = (pnl / initial_bankroll) * 100 if initial_bankroll > 0 else 0
        color = GREEN if pnl >= 0 else RED
        ax.annotate(
            f"  {final:.0f} EUR ({pnl:+.0f}, {pnl_pct:+.1f}%)",
            xy=(len(y) - 1, final),
            fontsize=11, fontweight="bold", color=color,
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Wetten", fontsize=10)
    ax.set_ylabel("EUR", fontsize=10)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_winloss_pie(
    wins: int,
    losses: int,
    open_bets: int = 0,
) -> bytes:
    """Generate a win/loss/open pie chart. Returns PNG bytes."""
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    labels = []
    sizes = []
    colors = []
    if wins > 0:
        labels.append(f"Won ({wins})")
        sizes.append(wins)
        colors.append(GREEN)
    if losses > 0:
        labels.append(f"Lost ({losses})")
        sizes.append(losses)
        colors.append(RED)
    if open_bets > 0:
        labels.append(f"Open ({open_bets})")
        sizes.append(open_bets)
        colors.append(GOLD)

    if not sizes:
        labels = ["No Data"]
        sizes = [1]
        colors = [GRID_COLOR]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.0f%%",
        startangle=90, textprops={"color": TEXT_COLOR, "fontsize": 12},
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontweight("bold")

    total = wins + losses
    hit_rate = (wins / total * 100) if total > 0 else 0
    ax.set_title(f"Hit Rate: {hit_rate:.0f}%", fontsize=14,
                 fontweight="bold", color=TEXT_COLOR, pad=20)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_sport_breakdown(
    sport_stats: Dict[str, Dict[str, float]],
) -> bytes:
    """Generate a per-sport ROI bar chart. Returns PNG bytes."""
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_theme(fig, ax)

    if not sport_stats:
        ax.text(0.5, 0.5, "Keine Daten", ha="center", va="center",
                fontsize=16, color=TEXT_COLOR, transform=ax.transAxes)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    sports = sorted(sport_stats.keys())
    rois = [sport_stats[s].get("roi", 0) * 100 for s in sports]
    bets = [int(sport_stats[s].get("bets", 0)) for s in sports]
    colors = [GREEN if r >= 0 else RED for r in rois]

    # Clean sport names for display
    display_names = [s.replace("_", " ").title()[:20] for s in sports]

    bars = ax.barh(display_names, rois, color=colors, edgecolor=DARK_BG, height=0.6)

    # Add value labels
    for bar, roi, n in zip(bars, rois, bets):
        w = bar.get_width()
        label = f" {roi:+.1f}% ({n})"
        ax.text(w + 0.5 if w >= 0 else w - 0.5, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left" if w >= 0 else "right",
                color=TEXT_COLOR, fontsize=10, fontweight="bold")

    ax.axvline(x=0, color=TEXT_COLOR, linewidth=1, alpha=0.5)
    ax.set_title("ROI nach Sport", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("ROI %", fontsize=10)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_dashboard(
    equity_curve: List[float],
    wins: int,
    losses: int,
    open_bets: int,
    initial_bankroll: float,
    pnl: float,
    roi: float,
    sport_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> bytes:
    """Generate a combined dashboard image with equity curve + pie + stats.

    Returns PNG bytes for a single premium image.
    """
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(DARK_BG)

    # Layout: top row = equity curve (wide), bottom row = pie + stats
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.3)

    # --- Top: Equity Curve ---
    ax_eq = fig.add_subplot(gs[0, :])
    _apply_dark_theme(fig, ax_eq)

    if equity_curve and len(equity_curve) > 1:
        x = list(range(len(equity_curve)))
        y = equity_curve
        ax_eq.fill_between(x, initial_bankroll, y,
                           where=[v >= initial_bankroll for v in y],
                           alpha=0.3, color=GREEN, interpolate=True)
        ax_eq.fill_between(x, initial_bankroll, y,
                           where=[v < initial_bankroll for v in y],
                           alpha=0.3, color=RED, interpolate=True)
        ax_eq.plot(x, y, color=GOLD, linewidth=2.5, zorder=5)
        ax_eq.axhline(y=initial_bankroll, color=TEXT_COLOR, linestyle="--", alpha=0.4)

        final = y[-1]
        color = GREEN if final >= initial_bankroll else RED
        ax_eq.annotate(
            f"  {final:.0f} EUR",
            xy=(len(y) - 1, final),
            fontsize=12, fontweight="bold", color=color,
        )
    else:
        ax_eq.text(0.5, 0.5, "Noch keine Daten", ha="center", va="center",
                   fontsize=14, color=TEXT_COLOR, transform=ax_eq.transAxes)

    ax_eq.set_title("Bankroll-Entwicklung", fontsize=13, fontweight="bold", pad=10)
    ax_eq.set_xlabel("Wetten", fontsize=9)
    ax_eq.set_ylabel("EUR", fontsize=9)

    # --- Bottom Left: Win/Loss Pie ---
    ax_pie = fig.add_subplot(gs[1, 0])
    ax_pie.set_facecolor(DARK_BG)

    total = wins + losses
    pie_data = []
    pie_labels = []
    pie_colors = []
    if wins > 0:
        pie_data.append(wins)
        pie_labels.append(f"Won ({wins})")
        pie_colors.append(GREEN)
    if losses > 0:
        pie_data.append(losses)
        pie_labels.append(f"Lost ({losses})")
        pie_colors.append(RED)
    if open_bets > 0:
        pie_data.append(open_bets)
        pie_labels.append(f"Open ({open_bets})")
        pie_colors.append(GOLD)

    if pie_data:
        ax_pie.pie(pie_data, labels=pie_labels, colors=pie_colors,
                   autopct="%1.0f%%", startangle=90,
                   textprops={"color": TEXT_COLOR, "fontsize": 10},
                   wedgeprops={"edgecolor": DARK_BG, "linewidth": 2})
    hit_rate = (wins / total * 100) if total > 0 else 0
    ax_pie.set_title(f"Hit Rate: {hit_rate:.0f}%", fontsize=12,
                     fontweight="bold", color=TEXT_COLOR, pad=10)

    # --- Bottom Right: Key Stats ---
    ax_stats = fig.add_subplot(gs[1, 1])
    ax_stats.set_facecolor(CARD_BG)
    ax_stats.axis("off")
    for spine in ax_stats.spines.values():
        spine.set_visible(False)

    current_br = equity_curve[-1] if equity_curve else initial_bankroll
    stats_text = (
        f"Bankroll:  {current_br:.2f} EUR\n"
        f"PnL:  {pnl:+.2f} EUR\n"
        f"ROI:  {roi * 100:+.1f}%\n"
        f"Wetten:  {total} (W:{wins} / L:{losses})\n"
        f"Offen:  {open_bets}"
    )
    ax_stats.text(0.1, 0.85, stats_text, transform=ax_stats.transAxes,
                  fontsize=13, color=TEXT_COLOR, fontfamily="monospace",
                  verticalalignment="top", linespacing=1.8)
    ax_stats.set_title("Statistiken", fontsize=12, fontweight="bold",
                       color=TEXT_COLOR, pad=10)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()
