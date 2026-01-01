#!/usr/bin/env python3
"""
데이터 재생성 스크립트
- Quantile 기반 정규화 적용
- 학습 가능한 데이터 생성
"""

import sys
sys.path.append('data')
from graph_builder import fix_data

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔄 데이터 재생성 시작")
    print("="*70)
    
    input_path = "etc/old_versions/processed_data_GNN_cpu.pkl"
    output_path = "data/processed_data/processed_data_GNN_v3.pkl"
    
    print(f"\n📂 Input:  {input_path}")
    print(f"📂 Output: {output_path}")
    print(f"🔧 Method: Quantile-based normalization")
    
    fix_data(input_path, output_path)
    
    print("\n" + "="*70)
    print("✅ 데이터 재생성 완료!")
    print("="*70)
    print(f"\n📋 다음 단계:")
    print(f"  python train_v2.py --data_path {output_path} --model graphsage --epochs 10")
