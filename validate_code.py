"""
코드 검증 스크립트 - 문법 및 Import 확인
"""

import sys
import pickle
import torch

print("="*80)
print("🔍 Code Validation")
print("="*80)
print()

# 1. Import 검증
print("1️⃣ Importing train_final...")
try:
    import train_final
    print("   ✅ Import successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# 2. 데이터 검증
print("\n2️⃣ Loading data...")
try:
    with open('data/processed_data/processed_data_GNN_v5.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"   ✅ Data loaded")
    print(f"      Users: {data['user'].num_nodes:,}")
    print(f"      Foods: {data['food'].num_nodes:,}")
    print(f"      Edges: {data[('user', 'eats', 'food')].edge_index.size(1):,}")
except Exception as e:
    print(f"   ❌ Data loading failed: {e}")
    sys.exit(1)

# 3. 모델 생성 검증
print("\n3️⃣ Creating model...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = train_final.HealthAwareRecommender(
        hidden_channels=32,
        out_channels=16,
        metadata=(data.node_types, data.edge_types),
        dropout=0.5,
        heads=2
    ).to(device)
    
    print(f"   ✅ Model created")
    print(f"      Device: {device}")
except Exception as e:
    print(f"   ❌ Model creation failed: {e}")
    sys.exit(1)

# 4. Forward pass 검증
print("\n4️⃣ Testing forward pass...")
try:
    # Prepare data
    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    
    # Sample edges
    edge_label_index = data[('user', 'eats', 'food')].edge_index[:, :100].to(device)
    
    # Forward
    with torch.no_grad():
        pred = model(x_dict, edge_index_dict, edge_label_index)
    
    # Count parameters after forward pass
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"   ✅ Forward pass successful")
    print(f"      Model parameters: {num_params:,}")
    print(f"      Output shape: {pred.shape}")
    print(f"      Output range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
    print(f"      Output mean: {pred.mean().item():.4f}")
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Loss 함수 검증
print("\n5️⃣ Testing loss function...")
try:
    criterion = train_final.HealthAwareLoss(
        lambda_health=0.01,
        ranking_weight=0.2
    )
    
    # Sample targets
    targets = torch.randint(0, 2, (100,), dtype=torch.float32).to(device)
    
    # Compute loss
    loss = criterion(pred, targets)
    
    print(f"   ✅ Loss computation successful")
    print(f"      Loss value: {loss.item():.4f}")
except Exception as e:
    print(f"   ❌ Loss computation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✅ All validations passed!")
print("="*80)
print("\n🎯 Code is ready to run!")
print("\nNext steps:")
print("  1. Windows에서 git pull origin main")
print("  2. python test_quick.py (빠른 테스트)")
print("  3. python train_final.py (전체 실험)")
print()
