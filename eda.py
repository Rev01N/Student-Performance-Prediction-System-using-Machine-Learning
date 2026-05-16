"""
eda.py
------
Exploratory Data Analysis — generates and saves all visualisation plots.

Plots generated:
  1.  Class distribution (bar chart)
  2.  Grade/result pie chart
  3.  Feature distributions (histograms)
  4.  Correlation heatmap
  5.  Box plots — feature vs Final_Result
  6.  Scatter — Study Hours vs Previous Marks (coloured by result)
  7.  Scatter — Attendance vs Previous Marks (coloured by result)
  8.  Pair plot (key features)
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "student_data.csv")
FIG_DIR    = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
PALETTE    = {"Pass": "#4CAF82", "Fail": "#E05C5C"}
BG_COLOR   = "#0F1117"
PANEL_COLOR= "#1A1D27"
TEXT_COLOR = "#E8EAF0"
GRID_COLOR = "#2A2D3A"
ACCENT     = "#7B68EE"

FEATURE_COLS = ["Study_Hours", "Attendance", "Previous_Marks",
                "Assignments", "Internal_Marks"]

def _style():
    """Apply dark theme globally."""
    plt.rcParams.update({
        "figure.facecolor"  : BG_COLOR,
        "axes.facecolor"    : PANEL_COLOR,
        "axes.edgecolor"    : GRID_COLOR,
        "axes.labelcolor"   : TEXT_COLOR,
        "axes.titlecolor"   : TEXT_COLOR,
        "xtick.color"       : TEXT_COLOR,
        "ytick.color"       : TEXT_COLOR,
        "text.color"        : TEXT_COLOR,
        "grid.color"        : GRID_COLOR,
        "grid.linewidth"    : 0.6,
        "legend.facecolor"  : PANEL_COLOR,
        "legend.edgecolor"  : GRID_COLOR,
        "font.family"       : "DejaVu Sans",
        "font.size"         : 11,
        "axes.titlesize"    : 14,
        "axes.labelsize"    : 12,
    })

def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [EDA] Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
def run_eda():
    _style()

    df = pd.read_csv(DATA_PATH)
    print(f"\n[EDA] Loaded {len(df)} rows from {DATA_PATH}")

    pass_colors = [PALETTE.get(r, ACCENT) for r in df["Final_Result"]]

    # ── 1. Class distribution bar chart ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["Final_Result"].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[k] for k in counts.index],
                  edgecolor=BG_COLOR, linewidth=1.5, width=0.5)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 4, str(v),
                ha="center", va="bottom", fontweight="bold", color=TEXT_COLOR)
    ax.set_title("Student Pass / Fail Distribution", pad=15)
    ax.set_xlabel("Final Result")
    ax.set_ylabel("Number of Students")
    ax.grid(axis="y", alpha=0.4)
    ax.set_ylim(0, counts.max() + 40)
    fig.tight_layout()
    _save(fig, "01_class_distribution.png")

    # ── 2. Pie chart ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    wedge_props = {"edgecolor": BG_COLOR, "linewidth": 2}
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index,
        colors=[PALETTE[k] for k in counts.index],
        autopct="%1.1f%%", startangle=140,
        wedgeprops=wedge_props, pctdistance=0.75,
        textprops={"color": TEXT_COLOR, "fontsize": 13}
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("Pass vs Fail — Proportion", pad=15)
    _save(fig, "02_pie_distribution.png")

    # ── 3. Feature histograms ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        ax = axes[i]
        ax.hist(df[col], bins=20, color=ACCENT, edgecolor=BG_COLOR,
                linewidth=0.8, alpha=0.85)
        ax.set_title(f"Distribution of {col.replace('_', ' ')}")
        ax.set_xlabel(col.replace("_", " "))
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)
    axes[-1].set_visible(False)
    fig.suptitle("Feature Distributions", fontsize=16, fontweight="bold",
                 color=TEXT_COLOR, y=1.02)
    fig.tight_layout()
    _save(fig, "03_feature_distributions.png")

    # ── 4. Correlation heatmap ────────────────────────────────────────────────
    df_enc = df.copy()
    df_enc["Result_Code"] = (df_enc["Final_Result"] == "Pass").astype(int)
    corr = df_enc[FEATURE_COLS + ["Result_Code"]].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.zeros_like(corr, dtype=bool)
    np.fill_diagonal(mask, True)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, mask=mask,
                linewidths=0.5, linecolor=BG_COLOR, ax=ax,
                annot_kws={"size": 10},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", pad=15)
    fig.tight_layout()
    _save(fig, "04_correlation_heatmap.png")

    # ── 5. Box plots — feature vs Final_Result ────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        ax = axes[i]
        groups = [df[df["Final_Result"] == r][col].values
                  for r in ["Pass", "Fail"]]
        bp = ax.boxplot(groups, labels=["Pass", "Fail"],
                        patch_artist=True, notch=False,
                        medianprops={"color": "#FFD700", "linewidth": 2.5},
                        whiskerprops={"color": TEXT_COLOR},
                        capprops={"color": TEXT_COLOR},
                        flierprops={"markerfacecolor": ACCENT,
                                    "markersize": 4, "alpha": 0.5})
        for patch, key in zip(bp["boxes"], ["Pass", "Fail"]):
            patch.set_facecolor(PALETTE[key])
            patch.set_alpha(0.75)
        ax.set_title(f"{col.replace('_', ' ')} by Result")
        ax.set_xlabel("Final Result")
        ax.set_ylabel(col.replace("_", " "))
        ax.grid(alpha=0.3, axis="y")
    axes[-1].set_visible(False)
    fig.suptitle("Feature Distribution by Pass / Fail", fontsize=16,
                 fontweight="bold", color=TEXT_COLOR, y=1.02)
    fig.tight_layout()
    _save(fig, "05_boxplots_by_result.png")

    # ── 6. Scatter: Study Hours vs Previous Marks ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for result, grp in df.groupby("Final_Result"):
        ax.scatter(grp["Study_Hours"], grp["Previous_Marks"],
                   c=PALETTE[result], label=result,
                   alpha=0.65, edgecolors="none", s=50)
    ax.set_title("Study Hours vs Previous Marks")
    ax.set_xlabel("Study Hours (per day)")
    ax.set_ylabel("Previous Marks")
    ax.legend(title="Result", framealpha=0.4)
    ax.grid(alpha=0.3)
    _save(fig, "06_scatter_study_vs_marks.png")

    # ── 7. Scatter: Attendance vs Previous Marks ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for result, grp in df.groupby("Final_Result"):
        ax.scatter(grp["Attendance"], grp["Previous_Marks"],
                   c=PALETTE[result], label=result,
                   alpha=0.65, edgecolors="none", s=50)
    ax.set_title("Attendance vs Previous Marks")
    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("Previous Marks")
    ax.legend(title="Result", framealpha=0.4)
    ax.grid(alpha=0.3)
    _save(fig, "07_scatter_attendance_vs_marks.png")

    # ── 8. Pair plot ──────────────────────────────────────────────────────────
    pair_df = df[["Study_Hours", "Attendance", "Previous_Marks",
                  "Internal_Marks", "Final_Result"]].copy()
    pair_df.rename(columns={
        "Study_Hours"   : "Study Hrs",
        "Attendance"    : "Attend %",
        "Previous_Marks": "Prev Marks",
        "Internal_Marks": "Internal",
        "Final_Result"  : "Result"
    }, inplace=True)

    g = sns.pairplot(pair_df, hue="Result",
                     palette={"Pass": "#4CAF82", "Fail": "#E05C5C"},
                     plot_kws={"alpha": 0.55, "s": 20, "edgecolor": "none"},
                     diag_kind="kde")
    g.figure.patch.set_facecolor(BG_COLOR)
    for ax_row in g.axes:
        for ax_ in ax_row:
            if ax_ is not None:
                ax_.set_facecolor(PANEL_COLOR)
                ax_.grid(color=GRID_COLOR, linewidth=0.5)
    g.figure.suptitle("Pair Plot — Key Features", y=1.02,
                       fontsize=14, color=TEXT_COLOR)
    _save(g.figure, "08_pairplot.png")

    print("\n[EDA] ✔ All plots saved to outputs/figures/\n")


if __name__ == "__main__":
    run_eda()
