"""
NutriGraphNet v2 — 빠른 검증 스크립트
1 fold, 30 epochs로 동작 확인
"""
import subprocess, sys

cmd = [
    sys.executable, "nutrigraphnet_v2.py",
    "--data_path", "data/processed_data/processed_data_GNN_v5.pkl",
    "--variants", "full",
    "--n_folds", "1",
    "--epochs", "30",
    "--hidden_channels", "64",
    "--out_channels", "32",
    "--heads", "2",
    "--num_layers", "2",
    "--output_dir", "results/quick_v2",
    "--print_every", "5",
]
print("Running quick validation...")
subprocess.run(cmd)
