"""Measure food-embedding collapse: NutriGraphNet vs HFRS-DA.

Hypothesis: NutriGraphNet's deep collaborative message passing over-smooths food
embeddings (they point in similar directions), which retrieves well (HR) but
orders badly (NDCG). HFRS-DA keeps food identity (ingredient/nutrition anchored)
so its embeddings stay spread out -> good NDCG.

Metrics per model's food embeddings F [n_food, d]:
  - mean pairwise cosine similarity of a random sample (higher = more collapsed)
  - effective rank via participation ratio of singular values
    PR = (sum s_i)^2 / sum(s_i^2), in [1, d]; low = collapsed to few directions
"""
import sys, pickle, torch, numpy as np
sys.path.insert(0, 'src')
import torch.nn.functional as F

torch.manual_seed(0); np.random.seed(0)
DEV = 'cpu'

def stats(name, Fmat):
    Fmat = Fmat.detach().float()
    n, d = Fmat.shape
    # sample for pairwise cosine
    idx = torch.randperm(n)[:4000]
    S = F.normalize(Fmat[idx], dim=1)
    cs = S @ S.T
    off = cs[~torch.eye(len(idx), dtype=torch.bool)]
    # participation ratio of singular values (effective rank)
    sv = torch.linalg.svdvals(Fmat - Fmat.mean(0, keepdim=True))
    pr = (sv.sum()**2 / (sv**2).sum()).item()
    print(f"  {name:24s}  meanCos={off.mean():.3f}  medCos={off.median():.3f}  "
          f"effRank(PR)={pr:5.1f}/{d}  ({100*pr/d:.0f}% of dims used)")

# ---- data ----
data = pickle.load(open('data/processed_data/processed_data_GNN_v5.pkl', 'rb'))
# strip to cpu tensors
x_dict = {k: data[k].x.float() for k in data.node_types if 'x' in data[k]}
ei_dict = {et: data[et].edge_index for et in data.edge_types}

print("=== Food-embedding collapse diagnostic (100% density, fold 1) ===\n")

# ---- HFRS-DA food embeddings ----
from nutrigraphnet_v2 import HFRSDAReal, NutriGraphNetV2
hf = HFRSDAReal(20820, 31458, 3284, emb_dim=64)
fi = data[('food','contains','ingredient')].edge_index
hf.attach_ingredients(fi, DEV)
hf.load_state_dict(torch.load('results/gpu/HFRSDA_real_100pct/hfrsda_real/fold_1/best_model.pth', map_location='cpu'))
hf.eval()
with torch.no_grad():
    all_food = torch.arange(31458)
    f_hf = hf._food_repr(all_food)   # NLA food representation actually used in scoring
stats("HFRS-DA (food repr)", f_hf)
stats("HFRS-DA (raw f_emb)", hf.f_emb.weight)

# ---- NutriGraphNet (full_cos) food embeddings after message passing ----
model = NutriGraphNetV2(hidden_channels=128, out_channels=64,
                        metadata=(list(data.node_types), list(data.edge_types)),
                        num_layers=3, heads=4, decoder_type='cosine')
# lazy-init GAT lin layers with one forward, then load weights
with torch.no_grad():
    try:
        _ = model.encoder(x_dict, ei_dict)
    except Exception as e:
        print("  (lazy init forward:", e, ")")
model.load_state_dict(torch.load('results/gpu/SOTA_full_cos_100pct/full_cos/fold_1/best_model.pth', map_location='cpu'), strict=False)
model.eval()
with torch.no_grad():
    z = model.encoder(x_dict, ei_dict)
stats("NutriGraphNet (food z)", z['food'])
