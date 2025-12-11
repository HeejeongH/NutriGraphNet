#!/usr/bin/env python3
"""
데이터 정규화 재적용
- 원본 데이터를 올바르게 정규화
"""

import torch
import pickle
import numpy as np

def quantile_normalize(weights):
    """Quantile 기반 정규화 - 상위 데이터 강조"""
    if isinstance(weights, torch.Tensor):
        weights_np = weights.cpu().numpy()
    else:
        weights_np = weights
    
    # 상위 10%를 1.0 근처로, 나머지는 0-0.9 범위로
    q90 = np.quantile(weights_np, 0.9)
    
    normalized = np.where(
        weights_np >= q90,
        0.9 + 0.1 * (weights_np - q90) / (weights_np.max() - q90 + 1e-8),
        0.9 * (weights_np / (q90 + 1e-8))
    )
    
    normalized = np.clip(normalized, 0, 1)
    return torch.tensor(normalized, dtype=torch.float32)


def minmax_normalize(weights):
    """MinMax 정규화 - 전체 범위를 0-1로"""
    if isinstance(weights, torch.Tensor):
        weights_np = weights.cpu().numpy()
    else:
        weights_np = weights
    
    if weights_np.max() > weights_np.min():
        normalized = (weights_np - weights_np.min()) / (weights_np.max() - weights_np.min())
    else:
        normalized = np.ones_like(weights_np) * 0.5
    
    return torch.tensor(normalized, dtype=torch.float32)


print("\n" + "="*70)
print("🔧 데이터 정규화 재적용")
print("="*70)

# Load original data
print("\n📂 Loading original data...")
with open('etc/old_versions/processed_data_GNN_cpu.pkl', 'rb') as f:
    data = pickle.load(f)

print("✅ Data loaded")

# Check original weights
eats_weight = data['user', 'eats', 'food'].edge_attr
print(f"\n📊 Original User-eats-Food weights:")
print(f"   Min: {eats_weight.min():.6f}")
print(f"   Max: {eats_weight.max():.6f}")
print(f"   Mean: {eats_weight.mean():.6f}")
print(f"   Median: {eats_weight.median():.6f}")

# Apply MinMax normalization (simpler, better for training)
print(f"\n🔧 Applying MinMax normalization...")
normalized_eats = minmax_normalize(eats_weight)

print(f"\n📊 Normalized User-eats-Food weights:")
print(f"   Min: {normalized_eats.min():.6f}")
print(f"   Max: {normalized_eats.max():.6f}")
print(f"   Mean: {normalized_eats.mean():.6f}")
print(f"   Median: {normalized_eats.median():.6f}")

# Apply to data
data['user', 'eats', 'food'].edge_attr = normalized_eats
data['food', 'rev_eats', 'user'].edge_attr = normalized_eats

# Normalize food-contains-ingredient
contains_weight = data['food', 'contains', 'ingredient'].edge_attr
normalized_contains = minmax_normalize(contains_weight)
data['food', 'contains', 'ingredient'].edge_attr = normalized_contains
data['ingredient', 'rev_contains', 'food'].edge_attr = normalized_contains

print(f"\n📊 Normalized Food-contains-Ingredient weights:")
print(f"   Min: {normalized_contains.min():.6f}")
print(f"   Max: {normalized_contains.max():.6f}")
print(f"   Mean: {normalized_contains.mean():.6f}")

# Save
output_path = "data/processed_data/processed_data_GNN_v3.pkl"
print(f"\n💾 Saving to: {output_path}")
with open(output_path, 'wb') as f:
    pickle.dump(data, f)

import os
file_size = os.path.getsize(output_path) / (1024 * 1024)
print(f"✅ Saved! File size: {file_size:.2f} MB")

print("\n" + "="*70)
print("✅ 정규화 완료!")
print("="*70)
print(f"\n📋 다음 단계:")
print(f"  python train_v2.py --data_path {output_path} --model graphsage --epochs 10")
