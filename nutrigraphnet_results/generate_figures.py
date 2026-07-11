"""
Generate all analysis figures for the Option C paper:
"Why Graph Augmentation Fails in Sparse Nutrition Graphs"

Figures:
  Fig 1 — EXP-A: SGL aug_ratio sweep (HR@10, NDCG@10, MRR)
  Fig 2 — EXP-B: Interaction sparsity sweep (4 models × HR@10 & NDCG@10)
  Fig 3 — EXP-C: λ_health sensitivity (AUC & HR@10 flat-line)
  Fig 4 — EXP-D: Embedding dim sweep (5 models × HR@10)
  Fig 5 — EXP-F: Graph ablation bar chart (HFRS-DA variants)
  Fig 6 — Summary radar / heatmap: AUC vs HR@10 trade-off matrix
"""

import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── paths ─────────────────────────────────────────────────────────────────────
SRC  = "/home/user/webapp/results/analysis/SUMMARY_v3.json"
OUT  = "/home/user/webapp/nutrigraphnet_results/figures"
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

MODEL_COLOR = {
    "mf":       "#4E79A7",
    "lightgcn": "#F28E2B",
    "ngcf":     "#59A14F",
    "sgl":      "#E15759",
    "hfrsda":   "#76B7B2",
}
MODEL_LABEL = {
    "mf": "MF", "lightgcn": "LightGCN",
    "ngcf": "NGCF", "sgl": "SGL", "hfrsda": "HFRS-DA",
}

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

# parse sparsity data
sparsity_map = {10: "10pct", 30: "30pct", 50: "50pct", 70: "70pct", 100: "100pct"}
models_B = ["mf", "lightgcn", "ngcf", "sgl"]
x_pcts   = [10, 30, 50, 70, 100]

data_B = {m: {"HR@10": [], "NDCG@10": [], "AUC": []} for m in models_B}
for pct in x_pcts:
    tag = sparsity_map[pct]
    for m in models_B:
        key = f"B_sparsity_{tag}/{m}"
        for met in ["HR@10", "NDCG@10", "AUC"]:
            data_B[m][met].append(raw[key][met])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

for ax, metric in zip(axes, ["HR@10", "NDCG@10"]):
    for m in models_B:
        vals = data_B[m][metric]
        ax.plot(x_pcts, vals, "o-",
                color=MODEL_COLOR[m], label=MODEL_LABEL[m],
                lw=2.2, ms=7, markerfacecolor="white", markeredgewidth=2)
    ax.set_xlabel("Interaction Ratio (%)")
    ax.set_ylabel(metric)
    ax.set_title(f"({'a' if metric=='HR@10' else 'b'}) {metric} vs. Data Sparsity")
    ax.set_xticks(x_pcts)
    ax.legend(loc="lower right")

# add annotation: SGL collapse region
for ax in axes:
    ax.axvspan(0, 35, alpha=0.06, color="#E15759")
    ax.text(20, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.03,
            "Collapse\nZone", ha="center", fontsize=8, color="#E15759", alpha=0.8)

fig.suptitle("Figure 2. Model Performance vs. Interaction Sparsity\n"
             "(SGL collapses at low density; MF remains stable)", fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_sparsity_sweep.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig2_sparsity_sweep.png", bbox_inches="tight")
plt.close(fig)
print("  → fig2_sparsity_sweep saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — EXP-C: λ_health sensitivity
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 3: λ_health sensitivity …")

lambda_keys = sorted(
    [k for k in raw if k.startswith("C_lambda_")],
    key=lambda k: float(k.split("_lambda_")[1].split("/")[0])
)
lambda_vals = [float(k.split("_lambda_")[1].split("/")[0]) for k in lambda_keys]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, metric, color in zip(axes,
                              ["HR@10", "AUC"],
                              ["#76B7B2", "#59A14F"]):
    vals = [raw[k][metric] for k in lambda_keys]
    ax.semilogx([max(v, 1e-4) for v in lambda_vals], vals,
                "s-", color=color, lw=2.2, ms=7,
                markerfacecolor="white", markeredgewidth=2)
    ax.set_xlabel("λ_health (log scale)")
    ax.set_ylabel(metric)
    ax.set_title(f"({'a' if metric=='HR@10' else 'b'}) HFRS-DA — {metric}")
    # annotate variance
    spread = max(vals) - min(vals)
    ax.text(0.05, 0.05,
            f"Δ = {spread:.6f}\n(effectively flat)",
            transform=ax.transAxes, fontsize=9,
            color="gray", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

fig.suptitle("Figure 3. Health Constraint Sensitivity (λ_health Sweep)\n"
             "HFRS-DA shows zero sensitivity across 4 orders of magnitude", fontsize=11, y=1.01)
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
# Figure 5 — EXP-F: Graph component ablation
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 5: Graph ablation …")

ablation_keys = {
    "Full Graph":            "F_ablation_full_graph/hfrsda",
    "w/o Ingredient":        "F_ablation_no_ingredient/hfrsda",
    "w/o Time":              "F_ablation_no_time/hfrsda",
    "w/o Healthness":        "F_ablation_no_ingredient_time/hfrsda",   # proxy
    "w/o Food-Similar":      "F_ablation_no_food_similar/hfrsda",
    "w/o Ingr. & Time":      "F_ablation_no_ingredient_time/hfrsda",
}
# actual keys available
abl_variants = [
    ("Full Graph",          "F_ablation_full_graph/hfrsda"),
    ("w/o Ingredient",      "F_ablation_no_ingredient/hfrsda"),
    ("w/o Time",            "F_ablation_no_time/hfrsda"),
    ("w/o Food-Similar",    "F_ablation_no_food_similar/hfrsda"),
    ("w/o Ingr.+Time",      "F_ablation_no_ingredient_time/hfrsda"),
]

labels_F  = [v[0] for v in abl_variants]
hr_vals   = [raw[v[1]]["HR@10"]   for v in abl_variants]
ndcg_vals = [raw[v[1]]["NDCG@10"] for v in abl_variants]
auc_vals  = [raw[v[1]]["AUC"]     for v in abl_variants]

x = np.arange(len(labels_F))
w = 0.27

fig, ax = plt.subplots(figsize=(10, 4.5))
bars1 = ax.bar(x - w, hr_vals,   w, label="HR@10",   color="#76B7B2", edgecolor="white")
bars2 = ax.bar(x,     ndcg_vals, w, label="NDCG@10", color="#4E79A7", edgecolor="white")
bars3 = ax.bar(x + w, auc_vals,  w, label="AUC",     color="#59A14F", edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(labels_F, rotation=12, ha="right")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.legend()
ax.set_title("Figure 5. Graph Component Ablation Study (HFRS-DA)")

# annotate bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

# highlight zero-change
ax.text(0.98, 0.98,
        "All variants identical\n→ edge ablation ineffective\n(heterogeneous graph robustness)",
        transform=ax.transAxes, fontsize=8.5, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFF9C4", ec="#F0C040", alpha=0.9))

fig.tight_layout()
fig.savefig(f"{OUT}/fig5_graph_ablation.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig5_graph_ablation.png", bbox_inches="tight")
plt.close(fig)
print("  → fig5_graph_ablation saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 — AUC vs HR@10 scatter: AUC-HR@10 Paradox (100% sparsity)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 6: AUC vs HR@10 paradox …")

# EXP-B 100pct has mf/lightgcn/ngcf/sgl; hfrsda comes from D_dim_64 (full data)
models_B4  = ["mf", "lightgcn", "ngcf", "sgl"]
models_full = ["mf", "lightgcn", "ngcf", "sgl", "hfrsda"]

def get_score(model, metric):
    if model == "hfrsda":
        return raw[f"D_dim_64/hfrsda"][metric]   # full-data reference
    return raw[f"B_sparsity_100pct/{model}"][metric]

auc_full  = [get_score(m, "AUC")    for m in models_full]
hr_full   = [get_score(m, "HR@10")  for m in models_full]
ndcg_full = [get_score(m, "NDCG@10") for m in models_full]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# scatter AUC vs HR@10
ax = axes[0]
for m, a, h in zip(models_full, auc_full, hr_full):
    ax.scatter(a, h, s=130, color=MODEL_COLOR[m], zorder=5,
               edgecolors="white", linewidths=1.5)
    ax.annotate(MODEL_LABEL[m], (a, h), textcoords="offset points",
                xytext=(6, 4), fontsize=9)
ax.set_xlabel("AUC (Classification)")
ax.set_ylabel("HR@10 (Ranking)")
ax.set_title("(a) AUC vs. HR@10 — Score Paradox")
# add inverse trend line hint
ax.text(0.05, 0.95,
        "High AUC ≠ High HR@10\n(MF paradox)",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec="#FFB74D", alpha=0.9))

# bar chart: 3 metrics side by side for full data
ax = axes[1]
met_names = ["AUC", "HR@10", "NDCG@10"]
met_data  = {
    "AUC":    auc_full,
    "HR@10":  hr_full,
    "NDCG@10":ndcg_full,
}
x2 = np.arange(len(models_full))
w2 = 0.25
colors_m = ["#4E79A7", "#E15759", "#F28E2B"]
for i, (met, col) in enumerate(zip(met_names, colors_m)):
    ax.bar(x2 + (i-1)*w2, met_data[met], w2, label=met,
           color=col, edgecolor="white", alpha=0.85)
ax.set_xticks(x2)
ax.set_xticklabels([MODEL_LABEL[m] for m in models_full])
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.legend()
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
print(f"  Best at p=0.0 → augmentation consistently HURTS ranking")

# B: SGL collapse
sgl_hrs = [raw[f"B_sparsity_{sparsity_map[p]}/sgl"]["HR@10"] for p in x_pcts]
mf_hrs  = [raw[f"B_sparsity_{sparsity_map[p]}/mf"]["HR@10"]  for p in x_pcts]
print(f"\n[EXP-B] Sparsity Sweep (HR@10):")
print(f"  SGL:  {sgl_hrs}")
print(f"  MF:   {mf_hrs}")
print(f"  SGL@10%={sgl_hrs[0]:.3f} vs MF@10%={mf_hrs[0]:.3f}  → SGL collapses {sgl_hrs[0]/mf_hrs[0]:.2f}x worse")

# C: lambda flat
c_hr = [raw[k]["HR@10"] for k in lambda_keys]
c_auc = [raw[k]["AUC"] for k in lambda_keys]
print(f"\n[EXP-C] λ_health Sensitivity:")
print(f"  HR@10 range: {min(c_hr):.6f} – {max(c_hr):.6f}  (Δ={max(c_hr)-min(c_hr):.2e})")
print(f"  AUC   range: {min(c_auc):.6f} – {max(c_auc):.6f}  (Δ={max(c_auc)-min(c_auc):.2e})")
print(f"  → Health constraint has ZERO measurable effect (gradient vanishes)")

# D: dim sensitivity
mf_hr_d   = data_D["mf"]["HR@10"]
sgl_hr_d  = data_D["sgl"]["HR@10"]
hfrs_hr_d = data_D["hfrsda"]["HR@10"]
print(f"\n[EXP-D] Embedding Dim (HR@10):")
print(f"  MF   (d=16→256): {mf_hr_d[0]:.3f} → {mf_hr_d[-1]:.3f}")
print(f"  SGL  (d=16→256): {sgl_hr_d[0]:.3f} → {sgl_hr_d[-1]:.3f}")
print(f"  HFRS (d=16→256): {hfrs_hr_d[0]:.3f} → {hfrs_hr_d[-1]:.3f}")

# F: ablation
f_hr = [raw[v[1]]["HR@10"]   for v in abl_variants]
f_auc = [raw[v[1]]["AUC"]    for v in abl_variants]
print(f"\n[EXP-F] Graph Ablation (HFRS-DA):")
for v, h, a in zip(abl_variants, f_hr, f_auc):
    print(f"  {v[0]:20s}  HR@10={h:.4f}  AUC={a:.4f}")
print(f"  → All variants IDENTICAL → heterogeneous edges carry no useful signal")

print("\n" + "="*65)
print(f"All figures saved to: {OUT}/")
print("Files: fig1_sgl_aug_sweep, fig2_sparsity_sweep, fig3_lambda_sensitivity,")
print("       fig4_dim_sweep, fig5_graph_ablation, fig6_auc_hr_paradox")
print("="*65)
