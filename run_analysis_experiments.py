"""
전면 보강된 논문용 분석 실험 스크립트
"NutriGraphNet: Health-Aware Food Recommendation on Heterogeneous Graphs"

실험 구성 (보강 버전):
  EXP-A: SGL aug_ratio sweep (0.0 ~ 0.5) — augmentation collapse 분석
  EXP-B: Graph sparsity vs performance (10%~100%) — MF paradox 분석 (HFRSDA 포함)
  EXP-C: λ_health sensitivity (0.0 ~ 1.0) — health loss 효과 (버그 수정 후)
  EXP-D: Embedding dimension sweep (16~256) — capacity 분석
  EXP-F: Graph component ablation
         - [v1] NutriGraphNet full model (EXP-F 원래 버전, 구조적 한계 있음)
         - [v2] NGCF 기반 실제 topology ablation (유효한 버전) ← NEW
  EXP-G: Layer depth sweep (num_layers 1~4, LightGCN/NGCF) — over-smoothing 분석 ← NEW

버그 수정 이력:
  - _get_food_health(): edge_attr=None → PyG EdgeStorage KeyError → try/except 처리
  - ablation에서 edge_attr=None 대신 edge_index=zeros() 사용
  - EXP-F: NGCF 기반 실제 propagation topology ablation 추가

Usage:
  python run_analysis_experiments.py --exp A      # SGL augmentation sweep
  python run_analysis_experiments.py --exp B      # Sparsity sweep (HFRSDA 포함)
  python run_analysis_experiments.py --exp C      # Lambda sweep (버그 수정 후)
  python run_analysis_experiments.py --exp D      # Embedding dim sweep
  python run_analysis_experiments.py --exp F      # Graph ablation (NGCF 기반)
  python run_analysis_experiments.py --exp G      # Layer depth sweep (NEW)
  python run_analysis_experiments.py --exp all    # 전체 (수 시간 소요)
  python run_analysis_experiments.py --quick      # 빠른 테스트 (1 fold, 30 epochs)
  python run_analysis_experiments.py --exp summary # 결과 요약만
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
    print(f"  CMD: {' '.join(cmd[-10:])}")  # 마지막 10개 args만 표시
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def exp_A_sgl_sweep(quick=False):
    """EXP-A: SGL aug_ratio 0.0 → 0.5 (6 steps)
    
    Research Question: 극희소 그래프(0.040%)에서 edge dropout이 왜 붕괴하는가?
    Hypothesis: aug_ratio=0.0 (no augmentation)이 최고 성능을 보임 → SGL collapse 증명
    """
    ratios  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    n_folds = 1 if quick else 5
    epochs  = 30  if quick else 300
    for r in ratios:
        run([
            "--variants",    "sgl",
            "--sgl_aug",     str(r),
            "--n_folds",     str(n_folds),
            "--epochs",      str(epochs),
            "--patience",    "10" if quick else "30",
            "--print_every", "5" if quick else "20",
        ], f"results/analysis/A_sgl_aug_{r:.1f}", f"SGL aug_ratio={r}")


def exp_B_sparsity_sweep(quick=False):
    """EXP-B: interaction sparsity 10%→100% — 전 모델 (HFRSDA 포함)
    
    보강 사항: HFRSDA 추가 (기존 EXP-B에는 HFRSDA 없었음)
    Research Question: sparsity가 증가할수록 각 모델이 어떻게 다르게 반응하는가?
    Expected: MF는 sparsity에 강인, GNN은 sparsity에 취약 (density < 0.040%)
    """
    ratios  = [0.1, 0.3, 0.5, 0.7, 1.0]
    # 보강: HFRSDA 추가
    models  = "mf,lightgcn,ngcf,sgl,hfrsda"
    n_folds = 1 if quick else 3
    epochs  = 30  if quick else 300
    for r in ratios:
        run([
            "--variants",          models,
            "--interaction_ratio", str(r),
            "--n_folds",           str(n_folds),
            "--epochs",            str(epochs),
            "--patience",          "10" if quick else "30",
            "--print_every",       "5" if quick else "20",
        ], f"results/analysis/B_sparsity_{int(r*100)}pct",
           f"Sparsity {int(r*100)}% — {models}")


def exp_C_lambda_sweep(quick=False):
    """EXP-C: λ_health 0.0 → 1.0 — NutriGraphNet full model
    
    버그 수정 이력: _get_food_health()에서 edge_attr=None crash → try/except 수정
    이제 health loss가 실제로 작동함. 이 실험으로 λ_health 최적값 확인.
    Research Question: λ_health가 health-aware 추천에 미치는 영향
    """
    lambdas = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    n_folds = 1 if quick else 5
    epochs  = 30  if quick else 300
    for lam in lambdas:
        run([
            "--variants",      "full",   # NutriGraphNet full model
            "--lambda_health", str(lam),
            "--n_folds",       str(n_folds),
            "--epochs",        str(epochs),
            "--patience",      "10" if quick else "30",
            "--print_every",   "5" if quick else "20",
        ], f"results/analysis/C_lambda_{lam}", f"λ_health={lam} (NutriGraphNet)")


def exp_D_dim_sweep(quick=False):
    """EXP-D: embedding dim 16→256 — 전 모델
    
    Research Question: embedding capacity가 성능에 미치는 영향
    기대: 희소 그래프에서는 작은 dim이 더 좋을 수 있음 (over-fitting 방지)
    """
    dims    = [16, 32, 64, 128, 256]
    models  = "mf,lightgcn,ngcf,sgl,hfrsda"
    n_folds = 1 if quick else 3
    epochs  = 30  if quick else 200
    for d in dims:
        run([
            "--variants",        models,
            "--out_channels",    str(d),
            "--hidden_channels", str(d * 2),
            "--n_folds",         str(n_folds),
            "--epochs",          str(epochs),
            "--patience",        "10" if quick else "30",
            "--print_every",     "5" if quick else "20",
        ], f"results/analysis/D_dim_{d}", f"emb_dim={d}")


def exp_F_graph_ablation(quick=False):
    """EXP-F: Graph component ablation — 두 가지 방식으로 실행

    [v1] NutriGraphNet full model (기존 방식)
         - 구조적 한계: HFRSDA는 edge_index를 사용하지 않음 → 결과 무효
         - 유지: NutriGraphNet (GNN 기반) 모델의 ablation은 유효

    [v2] NGCF 기반 실제 topology ablation (신규, 유효한 버전)
         - NGCF의 _propagate()는 train_ei를 실제로 사용
         - auxiliary edge와 연결된 food를 train_ei에서 제거 → 실제 propagation 경로 변경
         - 이를 통해 각 edge type의 기여도를 정확히 측정 가능

    Research Question: 어떤 auxiliary edge type이 GNN 성능에 가장 중요한가?
    """
    n_folds = 1 if quick else 5
    epochs  = 30  if quick else 300

    ablations = [
        ("full_graph",         []),
        ("no_ingredient",      ["--ablate_no_ingredient"]),
        ("no_time",            ["--ablate_no_time"]),
        ("no_healthness",      ["--ablate_no_healthness"]),
        ("no_food_similar",    ["--ablate_no_food_similar"]),
        ("no_ingredient_time", ["--ablate_no_ingredient", "--ablate_no_time"]),
        ("no_all_auxiliary",   ["--ablate_no_ingredient", "--ablate_no_time",
                                 "--ablate_no_food_similar"]),
    ]

    # v1: NutriGraphNet (full GNN model) — EXP-F 원래 방식
    print("\n  [EXP-F v1] NutriGraphNet ablation (full graph model)")
    for tag, extra in ablations:
        run([
            "--variants",    "full",
            "--n_folds",     str(n_folds),
            "--epochs",      str(epochs),
            "--patience",    "10" if quick else "30",
            "--print_every", "5" if quick else "20",
        ] + extra,
        f"results/analysis/F_v1_nutri_{tag}", f"[NutriGraphNet] Graph Ablation: {tag}")

    # v2: NGCF 기반 — 실제 propagation topology ablation (유효한 버전)
    print("\n  [EXP-F v2] NGCF topology ablation (valid propagation-based ablation)")
    for tag, extra in ablations:
        run([
            "--variants",        "ngcf",
            "--ablation_model",  "ngcf",   # EXP-F NGCF mode 활성화
            "--n_folds",         str(n_folds),
            "--epochs",          str(epochs),
            "--patience",        "10" if quick else "30",
            "--print_every",     "5" if quick else "20",
        ] + extra,
        f"results/analysis/F_v2_ngcf_{tag}", f"[NGCF] Topology Ablation: {tag}")

    # v2b: LightGCN 기반 — 비교용
    print("\n  [EXP-F v2b] LightGCN topology ablation")
    for tag, extra in ablations:
        run([
            "--variants",        "lightgcn",
            "--ablation_model",  "lightgcn",
            "--n_folds",         str(n_folds),
            "--epochs",          str(epochs),
            "--patience",        "10" if quick else "30",
            "--print_every",     "5" if quick else "20",
        ] + extra,
        f"results/analysis/F_v2b_lgcn_{tag}", f"[LightGCN] Topology Ablation: {tag}")


def exp_G_layer_depth_sweep(quick=False):
    """EXP-G: Layer depth sweep (num_layers 1~4) — LightGCN / NGCF

    신규 실험: over-smoothing 분석
    Research Question:
      - 극희소 그래프에서 레이어 수가 증가하면 성능이 개선/저하되는가?
      - MF ranking paradox의 원인: GNN이 과도한 message passing으로 표현력 손실?
    Hypothesis: num_layers=1이 희소 그래프에서 최적 (layer=3에서 over-smoothing 발생)
    Expected Finding: 이 결과로 논문 Section 5.3 "MF Paradox" 설명 강화
    """
    layer_counts = [1, 2, 3, 4]
    models  = "lightgcn,ngcf"
    n_folds = 1 if quick else 5
    epochs  = 30  if quick else 300

    for nl in layer_counts:
        run([
            "--variants",    models,
            "--num_layers",  str(nl),
            "--n_folds",     str(n_folds),
            "--epochs",      str(epochs),
            "--patience",    "10" if quick else "30",
            "--print_every", "5" if quick else "20",
        ], f"results/analysis/G_layers_{nl}",
           f"Layer Depth: num_layers={nl} ({models})")


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
        for model_key, v in data.items():
            agg = v.get("aggregated", {})
            summary[d.name + "/" + model_key] = {
                "AUC":      agg.get("auc",      {}).get("mean", None),
                "F1":       agg.get("f1",        {}).get("mean", None),
                "HR@10":    agg.get("HR@10",     {}).get("mean", None),
                "NDCG@10":  agg.get("NDCG@10",   {}).get("mean", None),
                "MRR":      agg.get("MRR",        {}).get("mean", None),
                "Health@10":agg.get("HealthGain@10",{}).get("mean", None),
            }

    print(f"\n{'Experiment':<50} {'AUC':>7} {'F1':>7} {'HR@10':>7} {'NDCG@10':>9} {'MRR':>7} {'HG@10':>7}")
    print("-" * 95)
    for k, m in summary.items():
        def fmt(v): return f"{v:.4f}" if v is not None else "  N/A "
        print(f"{k:<50} {fmt(m['AUC'])} {fmt(m['F1'])} {fmt(m['HR@10'])} "
              f"{fmt(m['NDCG@10'])} {fmt(m['MRR'])} {fmt(m['Health@10'])}")

    # Save summary JSON
    with open("results/analysis/SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Summary saved → results/analysis/SUMMARY.json")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="NutriGraphNet 논문용 분석 실험 runner (전면 보강 버전)"
    )
    ap.add_argument("--exp",   default="all",
                    choices=["A","B","C","D","F","G","all","summary"],
                    help="실행할 실험 선택")
    ap.add_argument("--quick", action="store_true",
                    help="Quick mode: 1 fold, 30 epochs (검증용)")
    args = ap.parse_args()

    q = args.quick
    if q:
        print("⚡ QUICK MODE: 1 fold, 30 epochs")

    print("\n" + "="*70)
    print("  NutriGraphNet 전면 보강 실험 파이프라인")
    print("  버그 수정: _get_food_health() + ablation edge_attr + health logging")
    print("  신규 실험: EXP-F v2 (NGCF topology ablation) + EXP-G (layer sweep)")
    print("="*70)

    if args.exp in ("A", "all"):
        print("\n📊 EXP-A: SGL Augmentation Ratio Sweep")
        exp_A_sgl_sweep(q)

    if args.exp in ("B", "all"):
        print("\n📊 EXP-B: Graph Sparsity Sweep (HFRSDA 포함 보강)")
        exp_B_sparsity_sweep(q)

    if args.exp in ("C", "all"):
        print("\n📊 EXP-C: λ_health Sensitivity (버그 수정 후, NutriGraphNet)")
        exp_C_lambda_sweep(q)

    if args.exp in ("D", "all"):
        print("\n📊 EXP-D: Embedding Dimension Sweep")
        exp_D_dim_sweep(q)

    if args.exp in ("F", "all"):
        print("\n📊 EXP-F: Graph Component Ablation (v1: NutriGraphNet, v2: NGCF topology)")
        exp_F_graph_ablation(q)

    if args.exp in ("G", "all"):
        print("\n📊 EXP-G: Layer Depth Sweep (NEW: over-smoothing 분석)")
        exp_G_layer_depth_sweep(q)

    if args.exp != "summary":
        collect_and_print_summary()
    else:
        collect_and_print_summary()
