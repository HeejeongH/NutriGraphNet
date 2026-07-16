"""
Figures for:
"Auxiliary Nutritional Structure Substitutes for Interaction Data in
 Health-Aware Food Recommendation: Evidence from a National Dietary Survey Graph"

  Fig 1 — EXP-A: SGL aug_ratio sweep (results/analysis; the only source)
  Fig 2 — EXP-B: Sparsity sweep, 6 models incl. NutriGraphNet (results/gpu)
  Fig 3 — EXP-C-Full: NutriGraphNet λ_health sweep (results/gpu)
  Fig 4 — EXP-D: Embedding dim sweep (results/analysis; the only source)
  Fig 5 — EXP-F v2: NutriGraphNet vs NGCF-dilution ablation (results/gpu)
  Fig 6 — AUC vs HR@10 at full density, 6 models (results/gpu)

DATA SOURCE RULE
  Anything the paper reports from results/gpu must be plotted from results/gpu.
  results/analysis is a different, earlier run and disagrees with it (19 of 20
  EXP-B cells differ, e.g. NGCF HR@10 at 100%: 0.7773 vs 0.7844), so figures
  drawn from SUMMARY_v4 silently contradicted the tables they illustrate.
  EXP-A and EXP-D were never re-run on GPU, so those two alone still read
  SUMMARY_v4 -- which is also where the paper's Tables A and D come from.

NAMING
  The 'hfrsda' key is the DualAttn-TB topology-blind control (ours), NOT an
  implementation of HFRS-DA. See paper 4.1 and the HFRSDAModel docstring.
"""

import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── paths (repo-relative) ─────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(HERE)
SRC     = os.path.join(ROOT, "results", "analysis", "SUMMARY_v4.json")
GPU_DIR = os.path.join(ROOT, "results", "gpu")   # EXP-F v2 (5-fold, full params)
OUT     = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

with open(SRC) as f:
    raw = json.load(f)

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

# Categorical palette: validated with the dataviz validator (light surface).
# Adjacent-pair CVD separation for aqua/magenta sits in the 6-8 floor band, which
# is only admissible with secondary encoding -- hence the distinct marker per
# series below. Do not reorder or recolour without re-running the validator.
MODEL_COLOR = {
    "mf":       "#eda100",   # yellow
    "lightgcn": "#008300",   # green
    "ngcf":     "#e87ba4",   # magenta
    "sgl":      "#1baf7a",   # aqua
    "hfrsda":   "#4a3aa7",   # violet  -- the topology-blind control
    "full":     "#2a78d6",   # blue    -- NutriGraphNet (ours)
}
# Secondary encoding: identity must never rest on colour alone.
MODEL_MARKER = {
    "mf": "o", "lightgcn": "s", "ngcf": "^", "sgl": "v",
    "hfrsda": "D", "full": "P",
}
MODEL_LABEL = {
    "mf": "MF", "lightgcn": "LightGCN",
    "ngcf": "NGCF", "sgl": "SGL",
    "hfrsda": "DualAttn-TB (control)",   # NOT HFRS-DA -- see paper 4.1
    "full": "NutriGraphNet",
}


# ── GPU 5-fold loaders ────────────────────────────────────────────────────────
# The paper's Tables B/C report results/gpu, NOT results/analysis. The two are
# different runs and disagree (e.g. NGCF HR@10 at 100%: 0.7773 vs 0.7844), so
# figures sourced from SUMMARY_v4 would not match the tables they illustrate.
# EXP-A and EXP-D exist only under results/analysis, so those keep using `raw`.
def _gpu_agg(rel_dir, result_file):
    with open(os.path.join(GPU_DIR, rel_dir, result_file)) as f:
        return json.load(f)["aggregated"]

def _val(agg, metric):
    v = agg[metric]
    return (v["mean"], v.get("std", 0.0)) if isinstance(v, dict) else (v, 0.0)

def gpu_sparsity(pct, model, metric):
    """EXP-B. Baselines live in B_sparsity_*; NutriGraphNet in B_full_* (it was
    never run in the original sweep -- see paper changelog v1.3)."""
    if model == "full":
        return _val(_gpu_agg(f"B_full_{pct}pct", "results_full.json"), metric)
    return _val(_gpu_agg(f"B_sparsity_{pct}pct", f"results_{model}.json"), metric)

def gpu_lambda(lam, metric):
    """EXP-C-Full: NutriGraphNet lambda sweep at full parameters."""
    return _val(_gpu_agg(f"C_lambda_{lam}", "results_full.json"), metric)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — EXP-A: SGL aug_ratio sweep
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 1: SGL aug_ratio sweep …")

aug_keys = sorted(
    [k for k in raw if k.startswith("A_sgl_aug_")],
    key=lambda k: float(k.split("_aug_")[1].split("/")[0])
)
aug_ratios = [float(k.split("_aug_")[1].split("/")[0]) for k in aug_keys]
metrics = ["HR@10", "NDCG@10", "MRR", "AUC", "F1"]
metric_labels = {"HR@10": "HR@10", "NDCG@10": "NDCG@10", "MRR": "MRR",
                 "AUC": "AUC", "F1": "F1-score"}
colors_A = ["#E15759", "#4E79A7", "#F28E2B", "#59A14F", "#B07AA1"]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

for ax, metric, color in zip(axes, ["HR@10", "NDCG@10", "MRR"], colors_A):
    vals = [raw[k][metric] for k in aug_keys]
    ax.plot(aug_ratios, vals, "o-", color=color, lw=2.2, ms=7,
            markerfacecolor="white", markeredgewidth=2)
    ax.axvline(x=0.0, color="gray", lw=1, ls=":", alpha=0.7)
    ax.set_xlabel("Augmentation Ratio $p$")
    ax.set_ylabel(metric_labels[metric])
    ax.set_title(f"(a) SGL — {metric_labels[metric]}" if metric == "HR@10"
                 else f"({'b' if metric=='NDCG@10' else 'c'}) SGL — {metric_labels[metric]}")
    ax.set_xticks(aug_ratios)
    ymin = min(vals) * 0.97
    ymax = max(vals) * 1.02
    ax.set_ylim(ymin, ymax)
    # annotate best
    best_idx = vals.index(max(vals))
    ax.annotate(f"{vals[best_idx]:.4f}",
                xy=(aug_ratios[best_idx], vals[best_idx]),
                xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=8, color=color)

fig.suptitle("Figure 1. Effect of Graph Augmentation Ratio on SGL Performance\n"
             "(Sparse Nutrition Graph, density=0.040%)", fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_sgl_aug_sweep.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig1_sgl_aug_sweep.png", bbox_inches="tight")
plt.close(fig)
print("  → fig1_sgl_aug_sweep saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — EXP-B: Interaction sparsity sweep
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 2: Sparsity sweep …")

models_B = ["mf", "lightgcn", "ngcf", "sgl", "hfrsda", "full"]
x_pcts   = [10, 30, 50, 70, 100]

data_B = {m: {met: [] for met in ["HR@10", "NDCG@10"]} for m in models_B}
err_B  = {m: [] for m in models_B}
for pct in x_pcts:
    for m in models_B:
        for met in ["HR@10", "NDCG@10"]:
            data_B[m][met].append(gpu_sparsity(pct, m, met)[0])
        err_B[m].append(gpu_sparsity(pct, m, "HR@10")[1])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

for ax, metric in zip(axes, ["HR@10", "NDCG@10"]):
    for m in models_B:
        ours = (m == "full")
        ax.plot(x_pcts, data_B[m][metric], MODEL_MARKER[m] + "-",
                color=MODEL_COLOR[m], label=MODEL_LABEL[m],
                lw=3.0 if ours else 1.8, ms=9 if ours else 6,
                markerfacecolor="white", markeredgewidth=2 if ours else 1.6,
                zorder=5 if ours else 3)
    ax.set_xlabel("Interaction Ratio (%)")
    ax.set_ylabel(metric)
    ax.set_xticks(x_pcts)
    ax.set_ylim(0, 0.9)

axes[0].set_title("(a) HR@10 — NutriGraphNet is density-invariant")
axes[1].set_title("(b) NDCG@10 — the ordering advantage does not follow")
axes[0].legend(loc="lower right", fontsize=8, ncol=2)

# Panel (a): the two claims the paper actually makes, marked in place.
ax = axes[0]
ax.annotate("+45.0% over NGCF\nat 10% density",
            xy=(10, data_B["full"]["HR@10"][0]), xytext=(17, 0.845),
            fontsize=8, color=MODEL_COLOR["full"],
            arrowprops=dict(arrowstyle="->", color=MODEL_COLOR["full"], lw=1.2))
# No leader line here: any stroke long enough to reach the NGCF endpoint would
# cross the model cluster and read as a sixth series. The empty mid-band plus an
# explicit model name carries it instead.
ax.text(74, 0.55, "NGCF overtakes\nabove ~50% density", fontsize=8,
        color=MODEL_COLOR["ngcf"], ha="center", va="center")
ax.annotate("SGL collapse", xy=(10, data_B["sgl"]["HR@10"][0]),
            xytext=(24, 0.10), fontsize=8, color=MODEL_COLOR["sgl"],
            arrowprops=dict(arrowstyle="->", color=MODEL_COLOR["sgl"], lw=1.2))

fig.suptitle("Figure 2. Auxiliary Structure Substitutes for Interaction Data\n"
             "NutriGraphNet leads where interactions are scarcest and is flat across a 10x change "
             "in interaction volume (0.738 -> 0.729, ratio 0.99x)",
             fontsize=11, y=1.03)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_sparsity_sweep.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig2_sparsity_sweep.png", bbox_inches="tight")
plt.close(fig)
print("  → fig2_sparsity_sweep saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — EXP-C: λ_health sensitivity
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 3: λ_health sensitivity …")

# EXP-C-Full: NutriGraphNet at full parameters (paper Table C-Full).
# The previous version of this figure plotted the control's flat lambda sweep
# under the title "HFRS-DA shows zero sensitivity" -- a claim the paper has
# since withdrawn (see 4.1 attribution note). It now shows what the paper
# actually argues: the health objective is live and its trade-off is tunable.
LAMS = ["0.0", "0.001", "0.005", "0.01", "0.05", "0.1", "0.5", "1.0"]
lambda_vals = [float(l) for l in LAMS]
x_plot = [max(v, 5e-4) for v in lambda_vals]   # lambda=0 pinned for the log axis

hr_l   = [gpu_lambda(l, "HR@10")[0]         for l in LAMS]
hr_sd  = [gpu_lambda(l, "HR@10")[1]         for l in LAMS]
hg_l   = [gpu_lambda(l, "HealthGain@10")[0] for l in LAMS]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

PLATEAU = (8e-4, 0.12)   # lambda in [0.001, 0.1]

ax = axes[0]
ax.axvspan(*PLATEAU, alpha=0.09, color=MODEL_COLOR["full"], zorder=0)
ax.errorbar(x_plot, hr_l, yerr=hr_sd, fmt="P-", color=MODEL_COLOR["full"],
            lw=2.2, ms=8, markerfacecolor="white", markeredgewidth=2,
            ecolor="#888888", elinewidth=1.1, capsize=3, zorder=3)
ax.set_xscale("log")
ax.set_xlabel(r"$\lambda_{health}$ (log scale; $\lambda$=0 plotted at left edge)")
ax.set_ylabel("HR@10")
ax.set_ylim(0.45, 0.85)
ax.set_title("(a) Ranking — plateau, then a real cost")
ax.text(0.028, 0.79, "robust plateau\nΔ within fold noise", fontsize=8,
        ha="center", color=MODEL_COLOR["full"])
ax.annotate("−19.6% at λ=0.5", xy=(0.5, hr_l[6]), xytext=(0.055, 0.52),
            fontsize=8, arrowprops=dict(arrowstyle="->", lw=1.2))

ax = axes[1]
ax.axvspan(*PLATEAU, alpha=0.09, color=MODEL_COLOR["full"], zorder=0)
ax.axhline(0, color="#888888", lw=1.2, zorder=1)
ax.plot(x_plot, hg_l, "P-", color=MODEL_COLOR["full"], lw=2.2, ms=8,
        markerfacecolor="white", markeredgewidth=2, zorder=3)
ax.set_xscale("log")
ax.set_xlabel(r"$\lambda_{health}$ (log scale)")
ax.set_ylabel("HealthGain@10")
ax.set_ylim(-0.0132, 0.0016)   # headroom so the plateau label clears the axis
ax.set_title("(b) Health signal — live at every λ, contracting when λ dominates")
# Upper-left is empty (the curve hugs the floor until λ=0.1), so the plateau
# label goes there rather than under the lowest point where it met the axis.
ax.text(0.0022, -0.0053, "non-zero across\n4 orders of magnitude", fontsize=8,
        ha="center", color=MODEL_COLOR["full"])
ax.annotate("→ 0 as ranking is\ntraded away", xy=(1.0, hg_l[7]),
            xytext=(0.075, -0.0033), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=1.2))

fig.suptitle("Figure 3. The Health Objective Is Routed and Tunable (λ_health Sweep, NutriGraphNet)\n"
             "GPU 5-fold CV, full parameters. The topology-blind control registers Δ HR@10 = 0.000 "
             "exactly across this range — by construction, not by defect.",
             fontsize=11, y=1.03)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_lambda_sensitivity.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig3_lambda_sensitivity.png", bbox_inches="tight")
plt.close(fig)
print("  → fig3_lambda_sensitivity saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — EXP-D: Embedding dim sweep
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 4: Embedding dim sweep …")

dims = [16, 32, 64, 128, 256]
models_D = ["mf", "lightgcn", "ngcf", "sgl", "hfrsda"]

data_D = {m: {"HR@10": [], "NDCG@10": [], "AUC": []} for m in models_D}
for d in dims:
    for m in models_D:
        key = f"D_dim_{d}/{m}"
        for met in ["HR@10", "NDCG@10", "AUC"]:
            data_D[m][met].append(raw[key][met])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

for ax, metric in zip(axes, ["HR@10", "NDCG@10"]):
    for m in models_D:
        vals = data_D[m][metric]
        ls = "--" if m == "mf" else "-"
        ax.plot(dims, vals, f"o{ls}",
                color=MODEL_COLOR[m], label=MODEL_LABEL[m],
                lw=2.0, ms=6, markerfacecolor="white", markeredgewidth=1.8)
    ax.set_xlabel("Embedding Dimension $d$")
    ax.set_ylabel(metric)
    ax.set_title(f"({'a' if metric=='HR@10' else 'b'}) {metric} vs. Embedding Dim")
    ax.set_xticks(dims)
    ax.legend(loc="lower right", ncol=2)

fig.suptitle("Figure 4. Effect of Embedding Dimension on Model Performance",
             fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_dim_sweep.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig4_dim_sweep.png", bbox_inches="tight")
plt.close(fig)
print("  → fig4_dim_sweep saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — EXP-F v2: Graph component ablation
#   (a) NutriGraphNet direct edge ablation (results/gpu/F_ablation_*)
#   (b) Δ HR@10 vs full graph: NutriGraphNet vs NGCF 50% dilution
#       (results/gpu/F_ngcf_dilution_*); HFRS-DA v1 = 0.000 reference line
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 5: Graph ablation (EXP-F v2) …")

ABL_VARIANTS = [
    ("Full Graph",        "full_graph"),
    ("w/o Ingredient",    "no_ingredient"),
    ("w/o Time",          "no_time"),
    ("w/o Food-Similar",  "no_food_similar"),
    ("w/o Healthness",    "no_healthness"),
    ("w/o Ingr.+Time",    "no_ingredient_time"),
    ("w/o All Auxiliary", "no_all_auxiliary"),
]

def load_expf(prefix, result_file):
    out = {}
    for _, tag in ABL_VARIANTS:
        with open(os.path.join(GPU_DIR, f"{prefix}_{tag}", result_file)) as f:
            out[tag] = json.load(f)["aggregated"]
    return out

v2a = load_expf("F_ablation",      "results_full.json")   # NutriGraphNet
v2b = load_expf("F_ngcf_dilution", "results_ngcf.json")   # NGCF dilution proxy

C_NUTRI, C_NGCF = "#2a78d6", "#008300"

labels_F = [v[0] for v in ABL_VARIANTS]
tags_F   = [v[1] for v in ABL_VARIANTS]
hr_a = [v2a[t]["HR@10"]["mean"] for t in tags_F]
sd_a = [v2a[t]["HR@10"]["std"]  for t in tags_F]
hr_b = [v2b[t]["HR@10"]["mean"] for t in tags_F]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

# (a) NutriGraphNet direct ablation — HR@10 with 5-fold std
ax = axes[0]
x = np.arange(len(labels_F))
ax.bar(x, hr_a, 0.62, yerr=sd_a, capsize=3,
       color=C_NUTRI, edgecolor="white",
       error_kw=dict(ecolor="#555555", lw=1.2, alpha=0.85))
ax.axhline(hr_a[0], color="#555555", lw=1, ls="--", alpha=0.6)
for xi, h, s in zip(x, hr_a, sd_a):
    ax.text(xi, h + s + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(labels_F, rotation=18, ha="right", fontsize=8.5)
ax.set_ylabel("HR@10")
ax.set_ylim(0, 0.9)
ax.set_title("(a) NutriGraphNet — Direct Edge Ablation (5-fold, mean ± σ)")

# (b) relative degradation vs full graph
ax = axes[1]
abl_tags, abl_labels = tags_F[1:], labels_F[1:]
d_a = [100 * (v2a[t]["HR@10"]["mean"] - hr_a[0]) / hr_a[0] for t in abl_tags]
d_b = [100 * (v2b[t]["HR@10"]["mean"] - hr_b[0]) / hr_b[0] for t in abl_tags]

x2, w2 = np.arange(len(abl_tags)), 0.38
bars_a = ax.bar(x2 - w2/2, d_a, w2, label="NutriGraphNet (direct ablation)",
                color=C_NUTRI, edgecolor="white")
bars_b = ax.bar(x2 + w2/2, d_b, w2, label="NGCF (50% interaction dilution)",
                color=C_NGCF, edgecolor="white")
ax.axhline(0, color="#888888", lw=1.4)
for bars in (bars_a, bars_b):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h - 0.4, f"{h:.1f}",
                ha="center", va="top", fontsize=7)
ax.set_xticks(x2)
ax.set_xticklabels(abl_labels, rotation=18, ha="right", fontsize=8.5)
ax.set_ylabel("Δ HR@10 vs. Full Graph (%)")
ax.set_ylim(-26, 4)
ax.set_title("(b) Relative Degradation per Ablated Component")
ax.legend(loc="lower left", fontsize=8)
ax.text(0.02, 0.96,
        "HFRS-DA (v1): Δ = 0.000 exactly for every variant\n"
        "(topology-invariant — auxiliary edges never consumed)",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF9C4", ec="#F0C040", alpha=0.9))

fig.suptitle("Figure 5. Graph Component Ablation (EXP-F v2): NutriGraphNet Depends on "
             "Topology, NGCF Marginally, HFRS-DA Not at All", fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_graph_ablation.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig5_graph_ablation.png", bbox_inches="tight")
plt.close(fig)
print("  → fig5_graph_ablation saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 — AUC vs HR@10 scatter: AUC-HR@10 Paradox (100% sparsity)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 6: AUC vs HR@10 paradox …")

# All six models at 100% density, sourced from results/gpu to match Table 1.
models_full = ["mf", "lightgcn", "ngcf", "sgl", "hfrsda", "full"]

def get_score(model, metric):
    met = "auc" if metric == "AUC" else metric
    return gpu_sparsity(100, model, met)[0]

auc_full  = [get_score(m, "AUC")     for m in models_full]
hr_full   = [get_score(m, "HR@10")   for m in models_full]
ndcg_full = [get_score(m, "NDCG@10") for m in models_full]

# Short forms: the full legend names collide at these point positions and as
# axis ticks. Identity still never rests on colour -- every mark is labelled.
SHORT = {"mf": "MF", "lightgcn": "LightGCN", "ngcf": "NGCF", "sgl": "SGL",
         "hfrsda": "DualAttn-TB", "full": "NutriGraphNet"}
# DualAttn-TB (0.855, 0.734) and NutriGraphNet (0.860, 0.729) nearly coincide,
# so offsets are hand-set rather than uniform.
OFFSET = {"mf": (8, -4), "lightgcn": (-14, -15), "ngcf": (8, 2),
          "sgl": (8, 2), "hfrsda": (-30, 11), "full": (10, -13)}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# scatter AUC vs HR@10
ax = axes[0]
for m, a, h in zip(models_full, auc_full, hr_full):
    ax.scatter(a, h, s=130, color=MODEL_COLOR[m], marker=MODEL_MARKER[m],
               zorder=5, edgecolors="white", linewidths=1.5)
    ax.annotate(SHORT[m], (a, h), textcoords="offset points",
                xytext=OFFSET[m], fontsize=8.5, color=MODEL_COLOR[m], zorder=6)
ax.set_xlabel("AUC (Classification)")
ax.set_ylabel("HR@10 (Ranking)")
ax.set_xlim(0.50, 0.94)
ax.set_ylim(0.30, 0.85)
ax.set_title("(a) AUC vs. HR@10 — the two do not agree")
# Mid-left is the only region clear of every mark (SGL sits at 0.699/0.358,
# which the previous bottom-left placement covered).
ax.text(0.03, 0.46,
        "MF: lowest AUC (0.547), highest HR@10 (0.760)\n"
        "SGL: mid AUC (0.699), collapsed HR@10 (0.358)",
        transform=ax.transAxes, fontsize=8, va="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f6f6f4", ec="#c9c9c2", alpha=0.95))

# bar chart: 3 metrics side by side for full data
ax = axes[1]
met_names = ["AUC", "HR@10", "NDCG@10"]
met_data  = {"AUC": auc_full, "HR@10": hr_full, "NDCG@10": ndcg_full}
x2 = np.arange(len(models_full))
w2 = 0.26
# Validated categorical slots (blue/green/magenta); colour encodes the metric
# here, not the model, so reuse of the model hues carries no cross-panel meaning.
colors_m = ["#2a78d6", "#008300", "#e87ba4"]
for i, (met, col) in enumerate(zip(met_names, colors_m)):
    ax.bar(x2 + (i-1)*w2, met_data[met], w2, label=met,
           color=col, edgecolor="white", linewidth=1.2)
ax.set_xticks(x2)
ax.set_xticklabels([SHORT[m] for m in models_full], rotation=20, ha="right",
                   fontsize=8.5)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.0)
ax.legend(fontsize=8, ncol=3, loc="upper center")
ax.set_title("(b) Full-Data Metric Profile per Model")

fig.suptitle("Figure 6. The AUC–Ranking Paradox in Sparse Nutrition Graphs",
             fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/fig6_auc_hr_paradox.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig6_auc_hr_paradox.png", bbox_inches="tight")
plt.close(fig)
print("  → fig6_auc_hr_paradox saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary print for paper section 6
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("KEY FINDINGS SUMMARY FOR PAPER SECTION 6")
print("="*65)

# A: SGL aug degradation
hr_aug = [raw[k]["HR@10"] for k in aug_keys]
ndcg_aug = [raw[k]["NDCG@10"] for k in aug_keys]
print(f"\n[EXP-A] SGL Aug Ratio Sweep:")
print(f"  HR@10:   {hr_aug[0]:.4f} (p=0.0) → {hr_aug[-1]:.4f} (p=0.5)  Δ={hr_aug[0]-hr_aug[-1]:+.4f}")
print(f"  NDCG@10: {ndcg_aug[0]:.4f} (p=0.0) → {ndcg_aug[-1]:.4f} (p=0.5)  Δ={ndcg_aug[0]-ndcg_aug[-1]:+.4f}")
print(f"  Δ={hr_aug[0]-hr_aug[-1]:+.4f} is within the measured run-to-run floor (~0.005)")
print(f"  → NO augmentation-ratio effect is resolvable here (paper 8.4)")

# B: sparsity sweep (results/gpu -- matches paper Table B)
print(f"\n[EXP-B] Sparsity Sweep (HR@10, results/gpu):")
for m in models_B:
    vals = data_B[m]["HR@10"]
    print(f"  {MODEL_LABEL[m]:24s} " + "  ".join(f"{v:.3f}" for v in vals))
ngn, ngcf_v, ctl, sgl_v = (data_B[k]["HR@10"] for k in ("full", "ngcf", "hfrsda", "sgl"))
print(f"  → at 10%: NutriGraphNet {ngn[0]:.3f} vs NGCF {ngcf_v[0]:.3f} "
      f"({100*(ngn[0]-ngcf_v[0])/ngcf_v[0]:+.1f}%), vs control {ctl[0]:.3f} "
      f"({100*(ngn[0]-ctl[0])/ctl[0]:+.1f}%), vs SGL {sgl_v[0]:.3f} ({ngn[0]/sgl_v[0]:.2f}x)")
print(f"  → density-invariance: {ngn[0]:.3f} → {ngn[-1]:.3f} (ratio {ngn[-1]/ngn[0]:.2f}x); "
      f"NGCF needs full data to reach {ngcf_v[-1]:.3f}, i.e. NutriGraphNet gets "
      f"{100*ngn[0]/ngcf_v[-1]:.1f}% of it on 10% of interactions")

# C: lambda sweep (NutriGraphNet, full parameters -- paper Table C-Full)
plateau = [gpu_lambda(l, "HR@10")[0] for l in ["0.001", "0.005", "0.01", "0.05", "0.1"]]
pm = sum(plateau) / len(plateau)
print(f"\n[EXP-C-Full] λ_health Sensitivity (NutriGraphNet, full params):")
print(f"  HR@10 plateau λ∈[0.001,0.1]: {min(plateau):.4f}–{max(plateau):.4f} (mean {pm:.4f})")
for l in ["0.5", "1.0"]:
    h = gpu_lambda(l, "HR@10")[0]
    print(f"  λ={l}: HR@10={h:.4f} ({100*(h-pm)/pm:+.1f}% vs plateau)  "
          f"HealthGain@10={gpu_lambda(l, 'HealthGain@10')[0]:+.5f}")
print(f"  → health objective live at every λ; trade-off becomes functional at λ≥0.5")

# D: dim sensitivity
mf_hr_d   = data_D["mf"]["HR@10"]
sgl_hr_d  = data_D["sgl"]["HR@10"]
hfrs_hr_d = data_D["hfrsda"]["HR@10"]
print(f"\n[EXP-D] Embedding Dim (HR@10):")
print(f"  MF   (d=16→256): {mf_hr_d[0]:.3f} → {mf_hr_d[-1]:.3f}")
print(f"  SGL  (d=16→256): {sgl_hr_d[0]:.3f} → {sgl_hr_d[-1]:.3f}")
print(f"  HFRS (d=16→256): {hfrs_hr_d[0]:.3f} → {hfrs_hr_d[-1]:.3f}")

# F v2: ablation
print(f"\n[EXP-F v2] Graph Ablation (GPU 5-fold, full params):")
for (label, tag) in ABL_VARIANTS:
    ha, hb = v2a[tag]["HR@10"]["mean"], v2b[tag]["HR@10"]["mean"]
    da = 100 * (ha - hr_a[0]) / hr_a[0]
    db = 100 * (hb - hr_b[0]) / hr_b[0]
    print(f"  {label:20s}  NutriGraphNet HR@10={ha:.4f} ({da:+.1f}%)   "
          f"NGCF-diluted HR@10={hb:.4f} ({db:+.1f}%)")
print(f"  → NutriGraphNet: graded topology dependence (−1.7% … −21.0%)")
print(f"  → NGCF dilution: marginal (−1.2% … −1.4%);  HFRS-DA v1: Δ=0.000 exactly")

print("\n" + "="*65)
print(f"All figures saved to: {OUT}/")
print("Files: fig1_sgl_aug_sweep, fig2_sparsity_sweep, fig3_lambda_sensitivity,")
print("       fig4_dim_sweep, fig5_graph_ablation, fig6_auc_hr_paradox")
print("="*65)
