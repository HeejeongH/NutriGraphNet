"""
Option C 논문용 추가 분석 실험 스크립트
"Why Graph Augmentation Fails in Sparse Nutrition Graphs"

실험 구성:
  EXP-A: SGL aug_ratio sweep (0.0 ~ 0.5)
  EXP-B: Graph sparsity vs performance (10%~100%)
  EXP-C: λ_health sensitivity (0.0 ~ 1.0)
  EXP-D: Embedding dimension sweep (16~256)
  EXP-E: Cold/Warm/Hot user 분석 (기존 결과 재분석)
  EXP-F: Graph component ablation (edge type 제거)

Usage:
  python run_analysis_experiments.py --exp A      # SGL augmentation sweep
  python run_analysis_experiments.py --exp B      # Sparsity sweep
  python run_analysis_experiments.py --exp C      # Lambda sweep
  python run_analysis_experiments.py --exp D      # Embedding dim sweep
  python run_analysis_experiments.py --exp F      # Graph ablation
  python run_analysis_experiments.py --exp all    # 전체 (수 시간 소요)
  python run_analysis_experiments.py --quick      # 빠른 테스트 (검증용)
"""

import subprocess, sys, json, argparse
from pathlib import Path

BASE_CMD = [
    sys.executable, "nutrigraphnet_v2.py",
    "--data_path", "data/processed_data/processed_data_GNN_v5.pkl",
    "--seed", "42",
]

def run(extra_args, out_dir, tag=""):
    """Run a single experiment variant."""
    cmd = BASE_CMD + extra_args + ["--output_dir", out_dir]
    print(f"\n{'='*60}")
    print(f"  Running: {tag}")
    print(f"  Output:  {out_dir}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def exp_A_sgl_sweep(quick=False):
    """EXP-A: SGL aug_ratio 0.0 → 0.5 (6 steps)"""
    ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    n_folds = 1 if quick else 5
    epochs  = 30  if quick else 300
    for r in ratios:
        run([
            "--variants", "sgl",
            "--sgl_aug",  str(r),
            "--n_folds",  str(n_folds),
            "--epochs",   str(epochs),
            "--patience", "10" if quick else "30",
            "--print_every", "5" if quick else "20",
        ], f"results/analysis/A_sgl_aug_{r:.1f}", f"SGL aug_ratio={r}")


def exp_B_sparsity_sweep(quick=False):
    """EXP-B: interaction sparsity 10%→100% (5 steps)
       모든 baseline 모델 비교
    """
    ratios  = [0.1, 0.3, 0.5, 0.7, 1.0]
    models  = "mf,lightgcn,ngcf,sgl"
    n_folds = 1 if quick else 3
    epochs  = 30  if quick else 300
    for r in ratios:
        run([
            "--variants",       models,
            "--interaction_ratio", str(r),   # nutrigraphnet_v2.py에 추가 필요
            "--n_folds",        str(n_folds),
            "--epochs",         str(epochs),
            "--patience",       "10" if quick else "30",
            "--print_every",    "5" if quick else "20",
        ], f"results/analysis/B_sparsity_{int(r*100)}pct",
           f"Sparsity {int(r*100)}% — {models}")


def exp_C_lambda_sweep(quick=False):
    """EXP-C: λ_health 0.0 → 1.0 (8 steps)"""
    lambdas = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    n_folds = 1 if quick else 5
    epochs  = 30  if quick else 300
    for lam in lambdas:
        run([
            "--variants",       "hfrsda",
            "--lambda_health",  str(lam),
            "--n_folds",        str(n_folds),
            "--epochs",         str(epochs),
            "--patience",       "10" if quick else "30",
            "--print_every",    "5" if quick else "20",
        ], f"results/analysis/C_lambda_{lam}", f"λ_health={lam}")


def exp_D_dim_sweep(quick=False):
    """EXP-D: embedding dim 16→256"""
    dims    = [16, 32, 64, 128, 256]
    models  = "mf,lightgcn,ngcf,sgl,hfrsda"
    n_folds = 1 if quick else 3
    epochs  = 30  if quick else 200
    for d in dims:
        run([
            "--variants",       models,
            "--out_channels",   str(d),
            "--hidden_channels",str(d * 2),
            "--n_folds",        str(n_folds),
            "--epochs",         str(epochs),
            "--patience",       "10" if quick else "30",
            "--print_every",    "5" if quick else "20",
        ], f"results/analysis/D_dim_{d}", f"emb_dim={d}")


def exp_F_graph_ablation(quick=False):
    """EXP-F: Graph component ablation
       full graph vs remove each edge type
    """
    n_folds = 1 if quick else 5
    epochs  = 30  if quick else 300

    ablations = [
        ("full_graph",          []),
        ("no_ingredient",       ["--ablate_no_ingredient"]),
        ("no_time",             ["--ablate_no_time"]),
        ("no_healthness",       ["--ablate_no_healthness"]),
        ("no_food_similar",     ["--ablate_no_food_similar"]),
        ("no_ingredient_time",  ["--ablate_no_ingredient", "--ablate_no_time"]),
    ]

    for tag, extra in ablations:
        run([
            "--variants",   "hfrsda",
            "--n_folds",    str(n_folds),
            "--epochs",     str(epochs),
            "--patience",   "10" if quick else "30",
            "--print_every","5" if quick else "20",
        ] + extra,
        f"results/analysis/F_ablation_{tag}", f"Graph Ablation: {tag}")


def collect_and_print_summary():
    """Collect all results and print summary table."""
    base = Path("results/analysis")
    if not base.exists():
        print("No results found yet.")
        return

    summary = {}
    for d in sorted(base.iterdir()):
        rf = d / "all_results.json"
        if not rf.exists():
            continue
        with open(rf) as f:
            data = json.load(f)
        # pick first model result
        for model_key, v in data.items():
            agg = v.get("aggregated", {})
            summary[d.name + "/" + model_key] = {
                "AUC":      agg.get("auc",      {}).get("mean", None),
                "F1":       agg.get("f1",        {}).get("mean", None),
                "HR@10":    agg.get("HR@10",     {}).get("mean", None),
                "NDCG@10":  agg.get("NDCG@10",   {}).get("mean", None),
                "MRR":      agg.get("MRR",        {}).get("mean", None),
            }

    print(f"\n{'Experiment':<45} {'AUC':>7} {'F1':>7} {'HR@10':>7} {'NDCG@10':>9} {'MRR':>7}")
    print("-" * 85)
    for k, m in summary.items():
        def fmt(v): return f"{v:.4f}" if v else "  N/A "
        print(f"{k:<45} {fmt(m['AUC'])} {fmt(m['F1'])} {fmt(m['HR@10'])} {fmt(m['NDCG@10'])} {fmt(m['MRR'])}")

    # Save summary JSON
    with open("results/analysis/SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Summary saved → results/analysis/SUMMARY.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp",   default="all",
                    choices=["A","B","C","D","F","all","summary"])
    ap.add_argument("--quick", action="store_true",
                    help="Quick mode: 1 fold, 30 epochs (for testing)")
    args = ap.parse_args()

    q = args.quick
    if q:
        print("⚡ QUICK MODE: 1 fold, 30 epochs")

    if args.exp in ("A", "all"):
        print("\n📊 EXP-A: SGL Augmentation Ratio Sweep")
        exp_A_sgl_sweep(q)

    if args.exp in ("B", "all"):
        print("\n📊 EXP-B: Graph Sparsity Sweep")
        exp_B_sparsity_sweep(q)

    if args.exp in ("C", "all"):
        print("\n📊 EXP-C: λ_health Sensitivity")
        exp_C_lambda_sweep(q)

    if args.exp in ("D", "all"):
        print("\n📊 EXP-D: Embedding Dimension Sweep")
        exp_D_dim_sweep(q)

    if args.exp in ("F", "all"):
        print("\n📊 EXP-F: Graph Component Ablation")
        exp_F_graph_ablation(q)

    if args.exp != "summary":
        collect_and_print_summary()
    else:
        collect_and_print_summary()
