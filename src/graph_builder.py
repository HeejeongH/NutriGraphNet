"""
기존 processed_data_GNN_cpu.pkl의 문제점 수정
- Edge weight 정규화
- Edge 이름 수정
- Health score 검증
"""

import torch
import pickle
import numpy as np
from pathlib import Path
import sys
import os

# Health score calculator import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def normalize_weights(weights, method='quantile'):
    """Edge weight 정규화
    
    Args:
        weights: 원본 가중치
        method: 정규화 방법
            - 'quantile': 상위 10%를 1.0으로, 나머지는 0-0.9 범위로 매핑 (권장)
            - 'minmax': 전체 범위를 0-1로 선형 매핑
            - 'log1p': Log 변환 후 정규화 (작은 값 강조, 권장 안 함)
            - 'clip': 이상치 제거 후 정규화
    """
    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().numpy()
    
    if method == 'quantile':
        # 상위 10%를 강조하는 정규화
        q90 = np.quantile(weights, 0.9)
        normalized = np.where(
            weights >= q90,
            0.9 + 0.1 * (weights - q90) / (weights.max() - q90 + 1e-8),
            0.9 * (weights / (q90 + 1e-8))
        )
        normalized = np.clip(normalized, 0, 1)
    elif method == 'minmax':
        # Min-max normalization
        if weights.max() > weights.min():
            normalized = (weights - weights.min()) / (weights.max() - weights.min())
        else:
            normalized = np.ones_like(weights) * 0.5
    elif method == 'log1p':
        # Log transformation: log(1 + x) - 작은 값 강조
        normalized = np.log1p(weights)
        if normalized.max() > 0:
            normalized = normalized / normalized.max()
    elif method == 'clip':
        # Clipping outliers
        normalized = np.clip(weights, 0, 10)
        if normalized.max() > 0:
            normalized = normalized / 10.0
    else:
        normalized = weights
    
    return torch.tensor(normalized, dtype=torch.float32)


def fix_data(input_path, output_path):
    """데이터 수정"""
    
    print(f"\n{'='*70}")
    print(f"🔧 Fixing Existing Graph Data")
    print(f"{'='*70}")
    
    # 데이터 로드
    print(f"\n📂 Loading data from: {input_path}")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ Data loaded")
    
    # 현재 상태 확인
    print(f"\n{'='*70}")
    print("📊 Current Data Status")
    print(f"{'='*70}")
    
    print(f"\n✅ Nodes:")
    for node_type in data.node_types:
        print(f"   {node_type}: {data[node_type].num_nodes:,}")
    
    print(f"\n✅ Edges:")
    for edge_type in data.edge_types:
        edge_count = data[edge_type].edge_index.shape[1]
        has_attr = hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None
        
        if has_attr:
            attr = data[edge_type].edge_attr
            print(f"   {edge_type}: {edge_count:,} edges, weights [{attr.min():.4f}, {attr.max():.4f}]")
        else:
            print(f"   {edge_type}: {edge_count:,} edges")
    
    # 수정 작업
    print(f"\n{'='*70}")
    print("🔧 Applying Fixes")
    print(f"{'='*70}")
    
    # 1. User-eats-Food edge weight 정규화
    if ('user', 'eats', 'food') in data.edge_types:
        print(f"\n1️⃣ Normalizing User-eats-Food weights...")
        old_weights = data[('user', 'eats', 'food')].edge_attr
        
        if old_weights is not None:
            print(f"   Before: [{old_weights.min():.4f}, {old_weights.max():.4f}], mean={old_weights.mean():.4f}")
            
            new_weights = normalize_weights(old_weights, method='log1p')
            data[('user', 'eats', 'food')].edge_attr = new_weights
            
            print(f"   After:  [{new_weights.min():.4f}, {new_weights.max():.4f}], mean={new_weights.mean():.4f}")
    
    # 2. Food-rev_eats-User edge weight도 동일하게
    if ('food', 'rev_eats', 'user') in data.edge_types:
        print(f"\n2️⃣ Normalizing Food-rev_eats-User weights...")
        old_weights = data[('food', 'rev_eats', 'user')].edge_attr
        
        if old_weights is not None:
            new_weights = normalize_weights(old_weights, method='log1p')
            data[('food', 'rev_eats', 'user')].edge_attr = new_weights
            print(f"   Updated: [{new_weights.min():.4f}, {new_weights.max():.4f}]")
    
    # 3. Food-contains-Ingredient edge weight 정규화
    if ('food', 'contains', 'ingredient') in data.edge_types:
        print(f"\n3️⃣ Normalizing Food-contains-Ingredient weights...")
        old_weights = data[('food', 'contains', 'ingredient')].edge_attr
        
        if old_weights is not None:
            print(f"   Before: [{old_weights.min():.4f}, {old_weights.max():.4f}]")
            
            new_weights = normalize_weights(old_weights, method='log1p')
            data[('food', 'contains', 'ingredient')].edge_attr = new_weights
            
            print(f"   After:  [{new_weights.min():.4f}, {new_weights.max():.4f}]")
    
    # 4. Ingredient-rev_contains-Food도 동일하게
    if ('ingredient', 'rev_contains', 'food') in data.edge_types:
        print(f"\n4️⃣ Normalizing Ingredient-rev_contains-Food weights...")
        old_weights = data[('ingredient', 'rev_contains', 'food')].edge_attr
        
        if old_weights is not None:
            new_weights = normalize_weights(old_weights, method='log1p')
            data[('ingredient', 'rev_contains', 'food')].edge_attr = new_weights
            print(f"   Updated: [{new_weights.min():.4f}, {new_weights.max():.4f}]")
    
    # 5. Edge 이름 변경: pairs -> similar (있으면)
    if ('food', 'pairs', 'food') in data.edge_types:
        print(f"\n5️⃣ Renaming Food-pairs-Food to Food-similar-Food...")
        
        # 데이터 복사
        data['food', 'similar', 'food'].edge_index = data[('food', 'pairs', 'food')].edge_index
        if hasattr(data[('food', 'pairs', 'food')], 'edge_attr'):
            data['food', 'similar', 'food'].edge_attr = data[('food', 'pairs', 'food')].edge_attr
        
        # 기존 삭제 (이건 직접 불가능하므로 무시)
        print(f"   ⚠️  Cannot delete old 'pairs' edge (PyG limitation)")
        print(f"   ✅ Added 'similar' edge with same data")
    
    # 6. Health score 확인
    if ('user', 'healthness', 'food') in data.edge_types:
        print(f"\n6️⃣ Checking Health scores...")
        health_scores = data[('user', 'healthness', 'food')].edge_attr
        
        if health_scores is not None:
            print(f"   Health scores: [{health_scores.min():.4f}, {health_scores.max():.4f}]")
            print(f"   Mean: {health_scores.mean():.4f}, Median: {health_scores.median():.4f}")
            
            # 이미 0-1 범위면 OK
            if health_scores.min() >= 0 and health_scores.max() <= 1.0:
                print(f"   ✅ Health scores are already normalized")
            else:
                print(f"   ⚠️  Health scores may need recalculation")
    
    # 최종 요약
    print(f"\n{'='*70}")
    print("📊 Fixed Data Summary")
    print(f"{'='*70}")
    
    print(f"\n✅ Edges:")
    for edge_type in data.edge_types:
        edge_count = data[edge_type].edge_index.shape[1]
        has_attr = hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None
        
        if has_attr:
            attr = data[edge_type].edge_attr
            print(f"   {edge_type}: {edge_count:,} edges, weights [{attr.min():.4f}, {attr.max():.4f}]")
        else:
            print(f"   {edge_type}: {edge_count:,} edges")
    
    # 저장
    print(f"\n💾 Saving fixed data to: {output_path}")
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✅ Data saved successfully!")
    
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nFile size: {file_size:.2f} MB")
    
    print(f"\n{'='*70}")
    print("🎉 All fixes applied!")
    print(f"{'='*70}\n")
    
    return data


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix existing graph data')
    parser.add_argument('--input', type=str, 
                       default='./processed_data/processed_data_GNN_cpu.pkl',
                       help='Input pickle file path')
    parser.add_argument('--output', type=str, 
                       default='./processed_data/processed_data_GNN_fixed.pkl',
                       help='Output pickle file path')
    
    args = parser.parse_args()
    
    fix_data(args.input, args.output)
