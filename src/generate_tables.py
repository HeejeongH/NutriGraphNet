# ================================================================
#  NutriGraphNet v2 — Paper Table Generator
#  Table 1: Main Comparison Results (5-fold CV)
#  Table 2: Detailed Per-Metric Full Table
# ================================================================

RESULTS = {
    "MF": {
        "type": "Non-graph",
        "f1":0.5436,"f1_s":0.0111,
        "auc":0.5468,"auc_s":0.0107,
        "ap":0.5389,"ap_s":0.0127,
        "hr5":0.6972,"hr5_s":0.0163,
        "ndcg5":0.5975,"ndcg5_s":0.0115,
        "hr10":0.7604,"hr10_s":0.0124,
        "ndcg10":0.6179,"ndcg10_s":0.0103,
        "hr20":0.8188,"hr20_s":0.0138,
        "ndcg20":0.6328,"ndcg20_s":0.0107,
        "mrr":0.5804,"mrr_s":0.0097,
        "prec":0.5326,"prec_s":0.0068,
        "rec":0.5551,"rec_s":0.0159,
    },
    "LightGCN": {
        "type": "Graph-based",
        "f1":0.6793,"f1_s":0.0006,
        "auc":0.8218,"auc_s":0.0012,
        "ap":0.8521,"ap_s":0.0013,
        "hr5":0.6124,"hr5_s":0.0214,
        "ndcg5":0.4634,"ndcg5_s":0.0153,
        "hr10":0.7208,"hr10_s":0.0223,
        "ndcg10":0.4986,"ndcg10_s":0.0157,
        "hr20":0.8052,"hr20_s":0.0236,
        "ndcg20":0.5201,"ndcg20_s":0.0155,
        "mrr":0.4386,"mrr_s":0.0129,
        "prec":0.5329,"prec_s":0.0006,
        "rec":0.9368,"rec_s":0.0015,
    },
    "NGCF": {
        "type": "Graph-based",
        "f1":0.7624,"f1_s":0.0056,
        "auc":0.8777,"auc_s":0.0014,
        "ap":0.8905,"ap_s":0.0028,
        "hr5":0.6928,"hr5_s":0.0194,
        "ndcg5":0.5269,"ndcg5_s":0.0233,
        "hr10":0.7844,"hr10_s":0.0209,
        "ndcg10":0.5569,"ndcg10_s":0.0222,
        "hr20":0.8460,"hr20_s":0.0146,
        "ndcg20":0.5725,"ndcg20_s":0.0209,
        "mrr":0.4915,"mrr_s":0.0238,
        "prec":0.6615,"prec_s":0.0106,
        "rec":0.8999,"rec_s":0.0042,
    },
    "SGL": {
        "type": "Self-supervised",
        "f1":0.6598,"f1_s":0.0019,
        "auc":0.6989,"auc_s":0.0027,
        "ap":0.7291,"ap_s":0.0028,
        "hr5":0.2640,"hr5_s":0.0154,
        "ndcg5":0.1982,"ndcg5_s":0.0115,
        "hr10":0.3576,"hr10_s":0.0234,
        "ndcg10":0.2283,"ndcg10_s":0.0140,
        "hr20":0.4852,"hr20_s":0.0215,
        "ndcg20":0.2606,"ndcg20_s":0.0135,
        "mrr":0.2089,"mrr_s":0.0109,
        "prec":0.6050,"prec_s":0.0014,
        "rec":0.7257,"rec_s":0.0026,
    },
    "HFRSDA": {
        "type": "Health-aware",
        "f1":0.7200,"f1_s":0.0164,
        "auc":0.8551,"auc_s":0.0103,
        "ap":0.8726,"ap_s":0.0201,
        "hr5":0.6730,"hr5_s":0.0266,
        "ndcg5":0.5780,"ndcg5_s":0.0182,
        "hr10":0.7340,"hr10_s":0.0222,
        "ndcg10":0.5977,"ndcg10_s":0.0152,
        "hr20":0.8010,"hr20_s":0.0233,
        "ndcg20":0.6149,"ndcg20_s":0.0127,
        "mrr":0.5635,"mrr_s":0.0142,
        "prec":0.5946,"prec_s":0.0309,
        "rec":0.9158,"rec_s":0.0217,
    },
}

# Best values per metric
METRICS = ["f1","auc","ap","hr5","ndcg5","hr10","ndcg10","hr20","ndcg20","mrr","prec","rec"]
best = {m: max(RESULTS[mdl][m] for mdl in RESULTS) for m in METRICS}

def bold(val, metric, fmt=".4f"):
    s = f"{val:{fmt}}"
    return f"\\textbf{{{s}}}" if abs(val - best[metric]) < 1e-9 else s

def fmt_cell(val, std):
    return f"{val:.4f}$_{{\\pm{std:.4f}}}$"

def bold_cell(val, std, metric):
    cell = fmt_cell(val, std)
    return f"\\textbf{{{cell}}}" if abs(val - best[metric]) < 1e-9 else cell

# ================================================================
# LaTeX TABLE 1: Compact Main Results
# ================================================================
latex_table1 = r"""% ==========================================================
%  Table 1: Comparison of Recommendation Models
%  Dataset: NutriGraphNet (20,820 users / 31,458 foods)
%  Protocol: 5-fold Cross-Validation (300 epochs / early-stop)
% ==========================================================
\begin{table*}[t]
\centering
\caption{Performance comparison of food recommendation models on the NutriGraphNet dataset
         (5-fold cross-validation, mean $\pm$ std).
         Bold values indicate the best performance per metric.
         $\dagger$ Health-aware variant; health loss term set to $\lambda_h=0.01$.}
\label{tab:main_results}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccccccc}
\toprule
\multirow{2}{*}{\textbf{Type}} &
\multirow{2}{*}{\textbf{Model}} &
\multicolumn{2}{c}{\textbf{Classification}} &
\multicolumn{3}{c}{\textbf{Ranking @5}} &
\multicolumn{3}{c}{\textbf{Ranking @10}} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
& & F1 & AUC & HR & NDCG & MRR & HR & NDCG & MRR \\
\midrule
"""

rows_compact = [
    ("MF",       "Non-graph"),
    ("LightGCN", "Graph-based"),
    ("NGCF",     "Graph-based"),
    ("SGL",      "Self-supervised"),
    ("HFRSDA",   "Health-aware$^\\dagger$"),
]

for model, mtype in rows_compact:
    d = RESULTS[model]
    row = (
        f"{mtype} & {model} & "
        f"{bold_cell(d['f1'],d['f1_s'],'f1')} & "
        f"{bold_cell(d['auc'],d['auc_s'],'auc')} & "
        f"{bold_cell(d['hr5'],d['hr5_s'],'hr5')} & "
        f"{bold_cell(d['ndcg5'],d['ndcg5_s'],'ndcg5')} & "
        f"{bold_cell(d['mrr'],d['mrr_s'],'mrr')} & "
        f"{bold_cell(d['hr10'],d['hr10_s'],'hr10')} & "
        f"{bold_cell(d['ndcg10'],d['ndcg10_s'],'ndcg10')} & "
        f"{bold_cell(d['mrr'],d['mrr_s'],'mrr')} \\\\"
    )
    latex_table1 += row + "\n"

latex_table1 += r"""\bottomrule
\end{tabular}
\vspace{1mm}
\footnotesize
\textit{Note}: HR = Hit Rate; NDCG = Normalized Discounted Cumulative Gain; MRR = Mean Reciprocal Rank.
All values are averaged over 5 folds. SGL uses contrastive self-supervised augmentation
but underperforms on ranking metrics due to graph augmentation collapse on sparse nutrition data.
\end{table*}
"""

# ================================================================
# LaTeX TABLE 2: Full Results (all metrics including @20)
# ================================================================
latex_table2 = r"""% ==========================================================
%  Table 2: Full Metric Results (All @k)
% ==========================================================
\begin{table*}[t]
\centering
\caption{Full evaluation results across all ranking cutoffs (5-fold CV, mean $\pm$ std).
         Bold = best per column.}
\label{tab:full_results}
\setlength{\tabcolsep}{3.5pt}
\small
\begin{tabular}{l|cc|cc|cc|cc|cc|c}
\toprule
\textbf{Model}
  & \textbf{F1} & \textbf{AUC}
  & \textbf{HR@5} & \textbf{NDCG@5}
  & \textbf{HR@10} & \textbf{NDCG@10}
  & \textbf{HR@20} & \textbf{NDCG@20}
  & \textbf{Prec} & \textbf{Rec}
  & \textbf{AP} \\
\midrule
"""

for model in ["MF","LightGCN","NGCF","SGL","HFRSDA"]:
    d = RESULTS[model]
    row = (
        f"{model} & "
        f"{bold_cell(d['f1'],d['f1_s'],'f1')} & "
        f"{bold_cell(d['auc'],d['auc_s'],'auc')} & "
        f"{bold_cell(d['hr5'],d['hr5_s'],'hr5')} & "
        f"{bold_cell(d['ndcg5'],d['ndcg5_s'],'ndcg5')} & "
        f"{bold_cell(d['hr10'],d['hr10_s'],'hr10')} & "
        f"{bold_cell(d['ndcg10'],d['ndcg10_s'],'ndcg10')} & "
        f"{bold_cell(d['hr20'],d['hr20_s'],'hr20')} & "
        f"{bold_cell(d['ndcg20'],d['ndcg20_s'],'ndcg20')} & "
        f"{bold_cell(d['prec'],d['prec_s'],'prec')} & "
        f"{bold_cell(d['rec'],d['rec_s'],'rec')} & "
        f"{bold_cell(d['ap'],d['ap_s'],'ap')} \\\\"
    )
    latex_table2 += row + "\n"

latex_table2 += r"""\bottomrule
\end{tabular}
\end{table*}
"""

# Save LaTeX
with open("nutrigraphnet_results/table1_main_results.tex", "w") as f:
    f.write(latex_table1)
with open("nutrigraphnet_results/table2_full_results.tex", "w") as f:
    f.write(latex_table2)

print("✅ LaTeX 테이블 생성 완료")
print("\n===== TABLE 1 =====")
print(latex_table1)
print("\n===== TABLE 2 =====")
print(latex_table2)
