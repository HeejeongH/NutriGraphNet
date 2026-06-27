"""
debug_ranking.py — HR@K=0 원인 진단 스크립트

실행: python debug_ranking.py
데이터 로드 → fast_link_split → 모델 encode → ranking 각 단계를 프린트
"""

import sys, pickle, numpy as np, torch
from pathlib import Path

# ── 0. 데이터 로드 ─────────────────────────────────────────────────────────────
DATA_DIR = Path("data/processed_data")
pkl_files = sorted(DATA_DIR.glob("*.pkl"))
if not pkl_files:
    print("ERROR: No pkl files found in data/processed_data/")
    sys.exit(1)

pkl_path = pkl_files[-1]
print(f"Loading: {pkl_path}")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print(f"\n[Data] node types: {data.node_types}")
print(f"[Data] edge types: {data.edge_types}")
for nt in data.node_types:
    n = data[nt].num_nodes
    x = data[nt].x.shape if hasattr(data[nt], 'x') and data[nt].x is not None else None
    print(f"  {nt}: num_nodes={n}, x={x}")

ei = data[('user','eats','food')].edge_index
print(f"\n[Edges] ('user','eats','food') edge_index shape: {ei.shape}")
print(f"  user index range: [{ei[0].min().item()}, {ei[0].max().item()}]")
print(f"  food index range: [{ei[1].min().item()}, {ei[1].max().item()}]")
print(f"  user num_nodes:   {data['user'].num_nodes}")
print(f"  food num_nodes:   {data['food'].num_nodes}")

# ── 1. fast_link_split 실행 ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from nutrigraphnet_v2 import fast_link_split

train_d, val_d, test_d = fast_link_split(data, val_ratio=0.05, test_ratio=0.10, seed=42)

def check_split(name, d):
    eil = d[('user','eats','food')].edge_label_index
    el  = d[('user','eats','food')].edge_label
    pos_mask = el == 1
    pos_eil  = eil[:, pos_mask]
    n_pos = pos_eil.shape[1]
    n_neg = (el == 0).sum().item()

    print(f"\n[{name}] edge_label_index shape: {eil.shape}")
    print(f"  pos={n_pos}, neg={n_neg}, ratio={n_pos/(n_pos+n_neg):.3f}")
    if n_pos > 0:
        print(f"  user idx range: [{pos_eil[0].min().item()}, {pos_eil[0].max().item()}]")
        print(f"  food idx range: [{pos_eil[1].min().item()}, {pos_eil[1].max().item()}]")
        print(f"  unique users with positives: {pos_eil[0].unique().shape[0]}")
    mp_ei = d[('user','eats','food')].edge_index
    print(f"  message-passing edge_index shape: {mp_ei.shape}")

check_split("train", train_d)
check_split("val",   val_d)
check_split("test",  test_d)

# ── 2. 모델 forward → z_dict 확인 ─────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[Device] {device}")

from nutrigraphnet_v2 import NutriGraphNetV2

model = NutriGraphNetV2(
    hidden_channels=64, out_channels=32,
    metadata=(train_d.node_types, train_d.edge_types),
    dropout=0.1, heads=2, drop_edge_p=0.0, num_layers=2,
).to(device)

# Lazy init
with torch.no_grad():
    _x  = {k: v.to(device) for k, v in train_d.x_dict.items()}
    _ei = {k: v.to(device) for k, v in train_d.edge_index_dict.items()}
    _eil = train_d[('user','eats','food')].edge_label_index[:, :16].to(device)
    model(_x, _ei, _eil)

# Encode on test data
with torch.no_grad():
    x_dict  = {k: v.to(device) for k, v in test_d.x_dict.items()}
    ei_dict = {k: v.to(device) for k, v in test_d.edge_index_dict.items()}
    z_dict  = model.encoder(x_dict, ei_dict)

print(f"\n[z_dict shapes]")
for k, v in z_dict.items():
    print(f"  {k}: {v.shape}")

print(f"\n[Embedding check]")
print(f"  z_dict['food'].shape[0] = {z_dict['food'].shape[0]}  (should == {data['food'].num_nodes})")
print(f"  z_dict['user'].shape[0] = {z_dict['user'].shape[0]}  (should == {data['user'].num_nodes})")

# ── 3. ranking_metrics_from_z 단계별 진단 ─────────────────────────────────────
print(f"\n[Ranking Diagnosis]")

eil_cpu = test_d[('user','eats','food')].edge_label_index.cpu()
el_cpu  = test_d[('user','eats','food')].edge_label.cpu()

pos_mask_np = (el_cpu.numpy() == 1)
eil_np = eil_cpu.numpy()
pos_users_np = eil_np[0, pos_mask_np]
pos_foods_np = eil_np[1, pos_mask_np]

print(f"  Total edge_label pairs: {eil_cpu.shape[1]}")
print(f"  Positive pairs:  {pos_users_np.shape[0]}")
print(f"  Unique pos users: {np.unique(pos_users_np).shape[0]}")
print(f"  pos food idx range: [{pos_foods_np.min()}, {pos_foods_np.max()}]")
print(f"  pos user idx range: [{pos_users_np.min()}, {pos_users_np.max()}]")

f_emb_np = z_dict['food'].detach().cpu().float().numpy()
u_emb_np = z_dict['user'].detach().cpu().float().numpy()
num_foods_emb = f_emb_np.shape[0]
num_users_emb = u_emb_np.shape[0]

print(f"\n  f_emb shape: {f_emb_np.shape}  (num_foods_emb={num_foods_emb})")
print(f"  u_emb shape: {u_emb_np.shape}  (num_users_emb={num_users_emb})")

# Check out-of-range
oor_users = (pos_users_np >= num_users_emb).sum()
oor_foods = (pos_foods_np >= num_foods_emb).sum()
print(f"\n  Out-of-range users (>= {num_users_emb}): {oor_users}")
print(f"  Out-of-range foods (>= {num_foods_emb}): {oor_foods}")

# ── 4. 샘플 사용자 1명으로 수동 확인 ─────────────────────────────────────────
unique_users = np.unique(pos_users_np)
u_idx = int(unique_users[0])
mask_u = (pos_users_np == u_idx)
pos_foods_u = pos_foods_np[mask_u]
pos_foods_u_valid = pos_foods_u[pos_foods_u < num_foods_emb]

print(f"\n[Sample user u_idx={u_idx}]")
print(f"  pos foods: {pos_foods_u.tolist()}")
print(f"  valid pos foods (<{num_foods_emb}): {pos_foods_u_valid.tolist()}")

if len(pos_foods_u_valid) > 0 and u_idx < num_users_emb:
    pos_set = set(int(f) for f in pos_foods_u_valid)
    u_vec  = u_emb_np[u_idx]
    scores = f_emb_np.dot(u_vec)
    sorted_idx = np.argsort(-scores)

    top20 = [int(fi) for fi in sorted_idx[:20]]
    print(f"  Top-20 food indices by score: {top20}")
    print(f"  pos_set: {pos_set}")
    print(f"  Intersection top-20 & pos_set: {set(top20) & pos_set}")
    print(f"  HR@5  = {1.0 if set([int(fi) for fi in sorted_idx[:5]]) & pos_set else 0.0}")
    print(f"  HR@10 = {1.0 if set([int(fi) for fi in sorted_idx[:10]]) & pos_set else 0.0}")
    print(f"  HR@20 = {1.0 if set([int(fi) for fi in sorted_idx[:20]]) & pos_set else 0.0}")

    # Score of pos foods
    for pf in list(pos_set)[:3]:
        rank = int(np.where(sorted_idx == pf)[0][0]) + 1
        print(f"  food {pf}: score={scores[pf]:.4f}, rank={rank}/{num_foods_emb}")

print("\n[DONE] — paste this output for diagnosis")
