#!/usr/bin/env python3
"""
역정규화 후 재정규화
Log1p로 압축된 데이터를 원래대로 복원한 후 MinMax로 재정규화
"""

import torch
import pickle
import numpy as np
from pathlib import Path

def reverse_log1p(normalized_weights, original_max=192.0):
    """
    Log1p 정규화를 역으로 풀어서 원본 값으로 복원
    normalized = log(1 + original) / log(1 + original_max)
    따라서: original = exp(normalized * log(1 + original_max)) - 1
    """
    log_max = np.log(1 + original_max)
    original = np.exp(normalized_weights * log_max) - 1
    return original

def normalize_minmax(weights):
    """MinMax 정규화 (0-1 범위)"""
    min_val = weights.min()
    max_val = weights.max()
    
    if max_val - min_val == 0:
        return torch.full_like(weights, 0.5)
    
    return (weights - min_val) / (max_val - min_val)

def fix_data_reverse_and_renormalize(input_path, output_path, original_max=192.0):
    """
    Log1p로 정규화된 데이터를 역정규화 후 MinMax로 재정규화
    """
    print("="*80)
    print("🔄 데이터 역정규화 및 재정규화")
    print("="*80)
    
    # 1. 데이터 로드
    print(f"\n📂 Loading data from: {input_path}")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # 2. User-eats-Food 엣지 처리
    edge_key = ('user', 'eats', 'food')
    if edge_key in data.edge_index_dict:
        edge_attr = data[edge_key].edge_attr
        
        print(f"\n📊 Original (Log1p normalized) Stats:")
        print(f"   Min: {edge_attr.min():.6f}")
        print(f"   Max: {edge_attr.max():.6f}")
        print(f"   Mean: {edge_attr.mean():.6f}")
        print(f"   Median: {edge_attr.median():.6f}")
        
        # 역정규화: Log1p → 원본 값으로 복원
        print(f"\n🔄 Reversing Log1p normalization (using original_max={original_max})...")
        edge_attr_numpy = edge_attr.numpy()
        original_weights = reverse_log1p(edge_attr_numpy, original_max)
        
        print(f"\n📊 Reversed (Original) Stats:")
        print(f"   Min: {original_weights.min():.6f}")
        print(f"   Max: {original_weights.max():.6f}")
        print(f"   Mean: {original_weights.mean():.6f}")
        print(f"   Median: {np.median(original_weights):.6f}")
        
        # MinMax 정규화
        print(f"\n✅ Applying MinMax normalization...")
        edge_attr_tensor = torch.from_numpy(original_weights).float()
        normalized_weights = normalize_minmax(edge_attr_tensor)
        
        print(f"\n📊 Final (MinMax normalized) Stats:")
        print(f"   Min: {normalized_weights.min():.6f}")
        print(f"   Max: {normalized_weights.max():.6f}")
        print(f"   Mean: {normalized_weights.mean():.6f}")
        print(f"   Median: {normalized_weights.median():.6f}")
        
        # 데이터 업데이트
        data[edge_key].edge_attr = normalized_weights
        
        # 역방향 엣지도 동일하게 처리
        rev_edge_key = ('food', 'rev_eats', 'user')
        if rev_edge_key in data.edge_index_dict:
            data[rev_edge_key].edge_attr = normalized_weights.clone()
    
    # 3. 데이터 저장
    print(f"\n💾 Saving to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved! File size: {file_size:.2f} MB")
    print("\n" + "="*80)
    print("✨ Data regeneration complete!")
    print("="*80)

if __name__ == "__main__":
    input_path = Path("data/processed_data/processed_data_GNN_fixed.pkl")
    output_path = Path("data/processed_data/processed_data_GNN_v4.pkl")
    
    if not input_path.exists():
        print(f"❌ Error: {input_path} not found!")
        exit(1)
    
    fix_data_reverse_and_renormalize(input_path, output_path)
    
    print("\n📝 Next step:")
    print("   python train_v2.py --data_path data/processed_data/processed_data_GNN_v4.pkl --model graphsage --epochs 10")
