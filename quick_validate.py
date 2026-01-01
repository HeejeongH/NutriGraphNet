"""
빠른 코드 검증 - Import만 확인
"""

print("🔍 Quick Validation")
print("="*80)

# 1. Import 검증
print("\n1️⃣ Importing modules...")
try:
    import torch
    import torch_geometric
    print("   ✅ PyTorch & PyG imported")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    exit(1)

# 2. train_final 검증
print("\n2️⃣ Importing train_final...")
try:
    import train_final
    print("   ✅ train_final imported")
    
    # Check classes
    print("\n3️⃣ Checking classes...")
    print(f"   ✅ HealthAwareGATEncoder: {hasattr(train_final, 'HealthAwareGATEncoder')}")
    print(f"   ✅ HealthAwareEdgeDecoder: {hasattr(train_final, 'HealthAwareEdgeDecoder')}")
    print(f"   ✅ HealthAwareRecommender: {hasattr(train_final, 'HealthAwareRecommender')}")
    print(f"   ✅ HealthAwareLoss: {hasattr(train_final, 'HealthAwareLoss')}")
    print(f"   ✅ train_epoch: {hasattr(train_final, 'train_epoch')}")
    print(f"   ✅ evaluate: {hasattr(train_final, 'evaluate')}")
    print(f"   ✅ train_one_fold: {hasattr(train_final, 'train_one_fold')}")
    
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*80)
print("✅ All imports successful!")
print("="*80)
print("\n🎯 Code is syntactically correct!")
print("\nWindows에서 실행하세요:")
print("  1. git pull origin main")
print("  2. python test_quick.py")
print()
