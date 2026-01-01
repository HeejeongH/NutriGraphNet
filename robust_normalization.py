#!/usr/bin/env python3
"""
Robust 정규화 - 이상치에 강한 정규화
Quantile 기반 clipping 후 MinMax 정규화
"""

import torch
import pickle
import numpy as np
from pathlib import Path

def robust_normalize(weights, lower_quantile=0.0, upper_quantile=0.95):
    """
    Robust normalization using quantile clipping
    
    Args:
        weights: PyTorch tensor of edge weights
        lower_quantile: Lower quantile for clipping (default: 0.0)
        upper_quantile: Upper quantile for clipping (default: 0.95, clips top 5%)
    """
    if isinstance(weights, torch.Tensor):
        weights_np = weights.cpu().numpy()
    else:
        weights_np = weights
    
    # Calculate quantile bounds
    lower_bound = np.quantile(weights_np, lower_quantile)
    upper_bound = np.quantile(weights_np, upper_quantile)
    
    print(f"   Quantile bounds: [{lower_bound:.6f}, {upper_bound:.6f}]")
    
    # Clip values
    clipped = np.clip(weights_np, lower_bound, upper_bound)
    
    # MinMax normalization on clipped values
    min_val = clipped.min()
    max_val = clipped.max()
    
    if max_val - min_val > 0:
        normalized = (clipped - min_val) / (max_val - min_val)
    else:
        normalized = np.ones_like(clipped) * 0.5
    
    return torch.tensor(normalized, dtype=torch.float32)


def fix_data_robust(input_path, output_path, upper_quantile=0.95):
    """
    Robust normalization - 상위 5% 이상치를 클리핑
    """
    print("="*80)
    print("🔧 Robust 정규화 (Quantile-based Clipping)")
    print("="*80)
    
    # 1. Load data
    print(f"\n📂 Loading data from: {input_path}")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # 2. Process User-eats-Food edges
    edge_key = ('user', 'eats', 'food')
    if edge_key in data.edge_index_dict:
        edge_attr = data[edge_key].edge_attr
        
        print(f"\n📊 Original Stats:")
        print(f"   Min:     {edge_attr.min():.6f}")
        print(f"   Max:     {edge_attr.max():.6f}")
        print(f"   Mean:    {edge_attr.mean():.6f}")
        print(f"   Median:  {edge_attr.median():.6f}")
        print(f"   95th percentile: {torch.quantile(edge_attr, 0.95):.6f}")
        print(f"   99th percentile: {torch.quantile(edge_attr, 0.99):.6f}")
        
        # Apply robust normalization
        print(f"\n🔧 Applying Robust normalization (clip at {upper_quantile*100:.0f}th percentile)...")
        normalized_weights = robust_normalize(edge_attr, 0.0, upper_quantile)
        
        print(f"\n📊 Normalized Stats:")
        print(f"   Min:     {normalized_weights.min():.6f}")
        print(f"   Max:     {normalized_weights.max():.6f}")
        print(f"   Mean:    {normalized_weights.mean():.6f}")
        print(f"   Median:  {normalized_weights.median():.6f}")
        print(f"   Std:     {normalized_weights.std():.6f}")
        
        # Distribution check
        print(f"\n📊 Distribution:")
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i in range(len(bins)-1):
            count = ((normalized_weights >= bins[i]) & (normalized_weights < bins[i+1])).sum()
            pct = count / len(normalized_weights) * 100
            print(f"   [{bins[i]:.1f}, {bins[i+1]:.1f}): {count:6d} ({pct:5.2f}%)")
        
        # Update data
        data[edge_key].edge_attr = normalized_weights
        
        # Update reverse edges
        rev_edge_key = ('food', 'rev_eats', 'user')
        if rev_edge_key in data.edge_index_dict:
            data[rev_edge_key].edge_attr = normalized_weights.clone()
    
    # 3. Process food-contains-ingredient (optional)
    contains_key = ('food', 'contains', 'ingredient')
    if contains_key in data.edge_index_dict:
        contains_attr = data[contains_key].edge_attr
        normalized_contains = robust_normalize(contains_attr, 0.0, upper_quantile)
        data[contains_key].edge_attr = normalized_contains
        
        rev_contains_key = ('ingredient', 'rev_contains', 'food')
        if rev_contains_key in data.edge_index_dict:
            data[rev_contains_key].edge_attr = normalized_contains.clone()
    
    # 4. Save
    print(f"\n💾 Saving to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved! File size: {file_size:.2f} MB")
    print("\n" + "="*80)
    print("✨ Robust normalization complete!")
    print("="*80)


if __name__ == "__main__":
    # Use the reversed data as input
    input_path = Path("data/processed_data/processed_data_GNN_v4.pkl")
    output_path = Path("data/processed_data/processed_data_GNN_v5.pkl")
    
    if not input_path.exists():
        print(f"❌ Error: {input_path} not found!")
        
        # Try using fixed data
        input_path = Path("data/processed_data/processed_data_GNN_fixed.pkl")
        if not input_path.exists():
            print(f"❌ Error: {input_path} not found either!")
            exit(1)
        
        print(f"⚠️ Using alternative: {input_path}")
    
    # Clip top 5% outliers (95th percentile)
    fix_data_robust(input_path, output_path, upper_quantile=0.95)
    
    print("\n📝 Next step:")
    print("   python train_v2.py --data_path data/processed_data/processed_data_GNN_v5.pkl --model graphsage --epochs 10")
