"""
Quick Test Script - 빠른 동작 확인용
1개 Fold, 10 Epochs로 빠르게 테스트
"""

import subprocess
import sys

print("🚀 Quick Test Starting...")
print("="*80)
print("Configuration:")
print("  - 1 Fold")
print("  - 10 Epochs")
print("  - Hidden Channels: 64")
print("  - Output Channels: 32")
print("="*80)
print()

cmd = [
    sys.executable, "train_final.py",
    "--data_path", "data/processed_data/processed_data_GNN_v5.pkl",
    "--n_folds", "1",
    "--epochs", "10",
    "--hidden_channels", "64",
    "--out_channels", "32",
    "--output_dir", "results/quick_test",
    "--print_every", "1"
]

subprocess.run(cmd)
