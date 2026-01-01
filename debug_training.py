#!/usr/bin/env python3
"""
학습 디버깅 - Gradient 흐름 및 Loss 계산 확인
"""

import torch
import torch.nn as nn
import pickle
from train_v2 import GraphSAGE_Model, prepare_train_test_split

print("="*70)
print("🔍 Training Debug")
print("="*70)

# 1. Load data
print("\n📂 Loading data...")
with open('data/processed_data/processed_data_GNN_v5.pkl', 'rb') as f:
    data = pickle.load(f)

# 2. Prepare train/test split
print("\n🔀 Preparing train/test split...")
train_data, test_data, threshold = prepare_train_test_split(data, test_ratio=0.2)

print(f"\n📊 Data info:")
print(f"   Train edges: {len(train_data['labels'])}")
print(f"   Threshold: {threshold:.3f}")
print(f"   Positive ratio: {train_data['labels'].float().mean():.3f}")

# 3. Create model
print("\n🏗️ Creating model...")
metadata = (list(data.node_types), list(data.edge_types))
input_dims = {
    'user': data['user'].x.size(1),
    'food': data['food'].x.size(1),
    'ingredient': data['ingredient'].x.size(1),
    'time': data['time'].x.size(1)
}

model = GraphSAGE_Model(
    hidden_channels=128,
    out_channels=64,
    metadata=metadata,
    input_dims=input_dims
)

# 4. Test forward pass
print("\n🔄 Testing forward pass...")
x_dict = data.x_dict
edge_index_dict = data.edge_index_dict

# Use a small batch
batch_size = 100
train_edge_index = train_data['edge_index'][:, :batch_size]
train_labels = train_data['labels'][:batch_size].float()

print(f"   Batch size: {batch_size}")
print(f"   Labels: {train_labels[:10]}")

output = model(x_dict, edge_index_dict, train_edge_index)

print(f"\n📊 Model output:")
print(f"   Shape: {output.shape}")
print(f"   Values: {output[:10]}")
print(f"   Min: {output.min():.6f}, Max: {output.max():.6f}")
print(f"   Mean: {output.mean():.6f}, Std: {output.std():.6f}")

# 5. Check if all values are the same
if output.std() < 1e-6:
    print("\n⚠️ WARNING: All predictions are identical!")
    print("   Model is not learning anything!")
else:
    print("\n✅ Predictions are diverse")

# 6. Test loss calculation
print("\n📉 Testing loss...")
criterion = nn.BCELoss()
loss = criterion(output, train_labels)
print(f"   Loss: {loss.item():.6f}")

# 7. Test backward pass
print("\n⬅️  Testing backward pass...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

optimizer.zero_grad()
loss.backward()

# Check gradients
has_gradients = False
total_grad_norm = 0.0
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        total_grad_norm += grad_norm
        if grad_norm > 1e-6:
            has_gradients = True
            print(f"   ✅ {name}: grad_norm = {grad_norm:.6f}")

if not has_gradients:
    print("\n❌ ERROR: No gradients!")
    print("   Model parameters are not being updated!")
else:
    print(f"\n✅ Total gradient norm: {total_grad_norm:.6f}")

# 8. Test optimizer step
print("\n🔄 Testing optimizer step...")
optimizer.step()

# Run forward again
output2 = model(x_dict, edge_index_dict, train_edge_index)
loss2 = criterion(output2, train_labels)

print(f"   Loss before: {loss.item():.6f}")
print(f"   Loss after:  {loss2.item():.6f}")
print(f"   Change: {(loss2.item() - loss.item()):.6f}")

if abs(loss2.item() - loss.item()) < 1e-6:
    print("\n❌ ERROR: Loss did not change!")
    print("   Optimizer is not updating parameters!")
else:
    print("\n✅ Loss changed - optimizer is working!")

print("\n" + "="*70)
print("✨ Debug complete!")
print("="*70)
