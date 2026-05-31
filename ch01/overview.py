"""
overview.py

LLM Landscape Visualization — pure matplotlib, no model loading.

Figure 1: Parameter count vs year (log scale, colored by architecture)
Figure 2: Architecture distribution pie chart
Figure 3: Japanese LLM timeline

Saved to model_landscape.png (3 subplots).
Markdown table printed to stdout.
"""

import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Model data
# ---------------------------------------------------------------------------
# Fields: name, year (float), params_B (billions), arch, notes

MODELS = [
    # name                      year    params_B   arch           notes
    ("GPT-1",                   2018.3,   0.117,  "decoder",     "First GPT, BooksCorpus"),
    ("BERT",                    2018.10,  0.110,  "encoder",     "Masked LM + NSP"),
    ("GPT-2",                   2019.2,   1.5,    "decoder",     "117M→1.5B, web text"),
    ("T5",                      2019.10,  11.0,   "enc-dec",     "Text-to-Text framework"),
    ("GPT-3",                   2020.5,  175.0,   "decoder",     "Few-shot learning"),
    ("PaLM",                    2022.4,  540.0,   "decoder",     "540B, Pathways"),
    ("ChatGPT",                 2022.11, 175.0,   "decoder+RLHF","GPT-3.5 + RLHF"),
    ("LLaMA",                   2023.2,   65.0,   "decoder",     "Open weights, Meta"),
    ("GPT-4",                   2023.3, 1000.0,   "decoder",     "Est. ~1T MoE"),
    ("Claude-2",                2023.7,   70.0,   "decoder+RLHF","Constitutional AI"),
    ("Mistral-7B",              2023.9,    7.0,   "decoder",     "Sliding window attn"),
    ("LLaMA-3",                 2024.4,   70.0,   "decoder",     "Meta, 8B/70B"),
    ("Gemini-1.5",              2024.2, 1000.0,   "decoder",     "Est. ~1T, 1M ctx"),
    ("Claude-3",                2024.3,  100.0,   "decoder+RLHF","Opus/Sonnet/Haiku"),
    ("Swallow-7B",              2024.1,    7.0,   "decoder",     "Japanese LLM, TokyoTech"),
    ("GPT-4o",                  2024.5,  200.0,   "multimodal",  "Omni: text/image/audio"),
]

# Japanese LLM timeline — separate dataset for Figure 3
JP_MODELS = [
    # name                      year    params_B    org
    ("rinna-GPT-2",             2021.6,   0.117,   "rinna"),
    ("CyberAgent-open-calm",    2022.11,   6.7,    "CyberAgent"),
    ("rinna-llama",             2023.5,   13.0,    "rinna"),
    ("StableLM-JP-3B",          2023.7,    3.6,    "Stability AI"),
    ("Swallow-7B",              2024.1,    7.0,    "Tokyo Tech"),
    ("ELYZA-japanese-Llama-2",  2023.8,   13.0,    "ELYZA"),
    ("Calm2-7B",                2023.11,   7.0,    "CyberAgent"),
    ("Japanese-StableLM-7B",    2023.9,    7.0,    "Stability AI"),
    ("Llama-3-Swallow-70B",     2024.7,   70.0,    "Tokyo Tech"),
    ("Qwen2-7B-Japanese",       2024.6,    7.0,    "Alibaba"),
    ("Sarashina2-70B",          2024.8,   70.0,    "SB Intuitions"),
    ("PLaMo-100B",              2024.9,  100.0,    "Preferred Networks"),
]

# ---------------------------------------------------------------------------
# Color scheme per architecture
# ---------------------------------------------------------------------------

ARCH_COLORS = {
    "decoder":      "#4C72B0",
    "encoder":      "#DD8452",
    "enc-dec":      "#55A868",
    "decoder+RLHF": "#C44E52",
    "multimodal":   "#8172B3",
}

ARCH_LABELS = {
    "decoder":      "Decoder-only",
    "encoder":      "Encoder-only",
    "enc-dec":      "Encoder-Decoder",
    "decoder+RLHF": "Decoder + RLHF",
    "multimodal":   "Multimodal",
}

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.patch.set_facecolor("#FAFAFA")

# ── Figure 1: Parameters vs Year ────────────────────────────────────────────

ax1 = axes[0]
ax1.set_facecolor("#F5F5F5")

for name, year, params_b, arch, notes in MODELS:
    color = ARCH_COLORS[arch]
    ax1.scatter(year, params_b, color=color, s=80, zorder=5,
                edgecolors="white", linewidths=0.8)
    # Label placement: offset to avoid overlap
    offset_x = 0.05
    offset_y = params_b * 1.35
    ax1.annotate(
        name,
        xy=(year, params_b),
        xytext=(year + offset_x, offset_y),
        fontsize=6.5,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        color="#333333",
    )

ax1.set_yscale("log")
ax1.set_xlabel("Year", fontsize=11)
ax1.set_ylabel("Parameters (Billions)", fontsize=11)
ax1.set_title("LLM Parameter Scale Over Time", fontsize=13, fontweight="bold")
ax1.set_xlim(2017.8, 2025.2)
ax1.set_ylim(0.05, 5000)
ax1.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}B")
)
ax1.grid(True, which="both", linestyle="--", alpha=0.4)

# Legend for architecture types
patches = [
    mpatches.Patch(color=c, label=ARCH_LABELS[a])
    for a, c in ARCH_COLORS.items()
]
ax1.legend(handles=patches, fontsize=8, loc="upper left",
           framealpha=0.9, title="Architecture")

# ── Figure 2: Architecture distribution pie ──────────────────────────────────

ax2 = axes[1]
arch_counts = {}
for _, _, _, arch, _ in MODELS:
    arch_counts[arch] = arch_counts.get(arch, 0) + 1

labels_pie = [ARCH_LABELS[a] for a in arch_counts]
sizes_pie  = list(arch_counts.values())
colors_pie = [ARCH_COLORS[a] for a in arch_counts]

wedges, texts, autotexts = ax2.pie(
    sizes_pie,
    labels=None,
    autopct="%1.0f%%",
    colors=colors_pie,
    startangle=140,
    pctdistance=0.75,
    wedgeprops=dict(linewidth=1.2, edgecolor="white"),
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_color("white")
    at.set_fontweight("bold")

ax2.legend(wedges, labels_pie, loc="lower center",
           bbox_to_anchor=(0.5, -0.18), fontsize=9, ncol=2,
           framealpha=0.9)
ax2.set_title("Architecture Distribution\n(16 notable models)",
              fontsize=13, fontweight="bold")

# ── Figure 3: Japanese LLM timeline ─────────────────────────────────────────

ax3 = axes[2]
ax3.set_facecolor("#F5F5F5")

ORG_COLORS = {
    "rinna":            "#E6194B",
    "CyberAgent":       "#3CB44B",
    "Stability AI":     "#4363D8",
    "Tokyo Tech":       "#F58231",
    "ELYZA":            "#911EB4",
    "Alibaba":          "#42D4F4",
    "SB Intuitions":    "#F032E6",
    "Preferred Networks":"#BFEF45",
}

for name, year, params_b, org in JP_MODELS:
    color = ORG_COLORS.get(org, "#888888")
    size  = max(30, min(300, params_b * 4))
    ax3.scatter(year, params_b, color=color, s=size, zorder=5,
                edgecolors="white", linewidths=0.8, alpha=0.9)
    ax3.annotate(
        name.replace("-", "\n", 1) if len(name) > 14 else name,
        xy=(year, params_b),
        xytext=(year + 0.05, params_b * 1.5),
        fontsize=6,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        color="#333333",
    )

ax3.set_yscale("log")
ax3.set_xlabel("Year", fontsize=11)
ax3.set_ylabel("Parameters (Billions)", fontsize=11)
ax3.set_title("Japanese LLM Timeline", fontsize=13, fontweight="bold")
ax3.set_xlim(2021.0, 2025.3)
ax3.set_ylim(0.05, 500)
ax3.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}B")
)
ax3.grid(True, which="both", linestyle="--", alpha=0.4)

org_patches = [
    mpatches.Patch(color=c, label=o)
    for o, c in ORG_COLORS.items()
    if any(org == o for _, _, _, org in JP_MODELS)
]
ax3.legend(handles=org_patches, fontsize=7, loc="upper left",
           framealpha=0.9, title="Organization", title_fontsize=8)

# ── Save ────────────────────────────────────────────────────────────────────

plt.tight_layout(pad=2.0)
out_path = os.path.join(_THIS_DIR, "model_landscape.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out_path}\n")

# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

COL_W = [22, 6, 12, 16, 35]
HEADERS = ["Model", "Year", "Params (B)", "Architecture", "Notes"]

def row(cells):
    return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, COL_W)) + " |"

def sep():
    return "|" + "|".join("-" * (w + 2) for w in COL_W) + "|"

lines = [row(HEADERS), sep()]
for name, year, params_b, arch, notes in sorted(MODELS, key=lambda x: x[1]):
    params_str = f"{params_b:g}B" if params_b < 1000 else f"~{params_b/1000:.0f}T"
    lines.append(row([name, f"{year:.0f}", params_str,
                       ARCH_LABELS.get(arch, arch), notes]))

table = "\n".join(lines)
print(table)
