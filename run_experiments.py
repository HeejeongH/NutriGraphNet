"""
NutriGraphNet v2 — Full Ablation + Baseline Experiment Script
논문 Table 2, 3 생성을 위한 완전 실험 실행 스크립트

실험 구성:
  1. Baselines:     MF, LightGCN
  2. Ablation:      no_dual, no_health, no_cl
  3. Full Model:    NutriGraphNet v2

Usage:
  python run_experiments.py                # 전체 실험 (5 folds x 300 epochs)
  python run_experiments.py --quick        # 빠른 테스트 (1 fold x 30 epochs)
"""

import subprocess
import sys
import json
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='Quick test mode')
    ap.add_argument('--output_dir', default='results/full_experiments')
    args_parsed = ap.parse_args()

    if args_parsed.quick:
        n_folds, epochs, hc, oc = 1, 30, 64, 32
        out = 'results/quick_ablation'
        heads, num_layers = 2, 2
    else:
        n_folds, epochs, hc, oc = 5, 300, 128, 64
        out = args_parsed.output_dir
        heads, num_layers = 4, 3

    variants = "full,no_health,no_cl,no_dual,mf,lightgcn"

    cmd = [
        sys.executable, "nutrigraphnet_v2.py",
        "--data_path", "data/processed_data/processed_data_GNN_v5.pkl",
        "--variants", variants,
        "--n_folds", str(n_folds),
        "--epochs", str(epochs),
        "--hidden_channels", str(hc),
        "--out_channels", str(oc),
        "--heads", str(heads),
        "--num_layers", str(num_layers),
        "--lambda_health", "0.1",
        "--lambda_cl", "0.05",
        "--temperature", "0.2",
        "--patience", str(30 if not args_parsed.quick else 10),
        "--output_dir", out,
        "--print_every", "10" if args_parsed.quick else "20",
    ]

    print("=" * 80)
    print("  NutriGraphNet v2 — Full Experiment Runner")
    print(f"  Mode: {'QUICK TEST' if args_parsed.quick else 'FULL EXPERIMENT'}")
    print(f"  Folds: {n_folds}, Epochs: {epochs}")
    print(f"  Variants: {variants}")
    print("=" * 80)
    print()

    subprocess.run(cmd)

    # Print summary
    result_file = Path(out) / "all_results.json"
    if result_file.exists():
        with open(result_file) as f:
            results = json.load(f)

        print("\n" + "=" * 80)
        print("  FINAL RESULTS SUMMARY")
        print("=" * 80)

        key_metrics = ['auc', 'f1', 'NDCG@10', 'HR@10', 'MRR', 'HealthGain@10']
        header = f"{'Model':<20}" + "".join(f"{m:>18}" for m in key_metrics)
        print(header)
        print("-" * len(header))

        model_order = ['mf', 'lightgcn', 'no_dual', 'no_health', 'no_cl', 'full']
        for v in model_order:
            if v not in results:
                continue
            agg = results[v].get('aggregated', {})
            row = f"{v:<20}"
            for m in key_metrics:
                if m in agg:
                    row += f"{agg[m]['mean']:>10.4f}±{agg[m]['std']:.3f}"
                else:
                    row += f"{'--':>18}"
            print(row)

        print("=" * 80)


if __name__ == '__main__':
    main()
