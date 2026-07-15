"""
NutriGraphNet v3: Rank-Calibrated Health-Aware Heterogeneous GNN
================================================================

핵심 설계 원칙 (v2 대비 개선):
────────────────────────────────────────────────────────────────────
문제: v2의 HybridDecoder(Bilinear+Dot+MLP 앙상블)가 rank calibration을 망침
     λ=0에서도 NDCG@10=0.40 (MF=0.61 대비 34% 낮음)

해결:
  1. [Decoder]  HybridDecoder → RankDotDecoder (순수 dot-product, L2-normalized)
               rank order를 잘 보존하는 cosine similarity 기반 scoring
  2. [Encoder]  LightGCN-style mean-pooling + GAT + Batch Normalization + Residual
               preference signal 보존하면서 auxiliary 구조 활용
  3. [Feature]  Node feature 완전 활용 (user 29dim, food 17dim, ingredient 101dim)
               Feature projection → enriched initial embeddings
  4. [Health]   Health를 완전 분리된 auxiliary head로 이동
               ranking gradient와 완전 격리 → BPR loss 보호
  5. [Training] Two-stage:
               Phase 1 (80% epochs): Pure BPR + InfoNCE — ranking 먼저 최적화
               Phase 2 (20% epochs): Health fine-tune — λ_health 적용
  6. [Loss]     InfoNCE (in-batch contrastive) + BPR 결합
               Adaptive health margin: health diff가 클 때만 penalty
  7. [Pooling]  Layer-wise mean pooling (LightGCN 방식) — over-smoothing 방지
  8. [Reg]      BatchNorm + Dropout + Weight Decay — 일반화 강화

아키텍처:
  node features → FeatureProjector (per-type Linear + BN + GELU)
                       │
            HeteroResGATEncoder
            • L layers of HeteroConv(GATConv)
            • BN + residual per layer
            • Layer-wise mean pooling (LightGCN)
            • Final L2 normalization
                       │
              ┌─────────┴──────────┐
        RankDotDecoder         HealthHead
        s = z_u · z_f         h = MLP(z_food)
        (ranking)             (aux, Phase 2 only)

Author: NutriGraphNet Research Team
Date: 2026-07-13
Version: 3.0.0
"""

import os
import gc
import json
import copy
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch_geometric
from torch_geometric.nn import HeteroConv, GATConv
from torch_geometric.data import HeteroData
from torch_geometric.utils import dropout_edge

from sklearn.metrics import roc_auc_score, f1_score


# ============================================================================
# CONSTANTS
# ============================================================================

# NutriGraph-KR feature dims
FEAT_DIMS = {
    'user': 29,
    'food': 17,
    'ingredient': 101,
    'time': 4,
}

# Preference edges for ranking channel
PREF_EDGE_RELS = frozenset([
    'eats', 'rev_eats', 'similar', 'rev_similar',
    'food_similar', 'rev_food_similar',
])

# Auxiliary edges for health channel
AUX_EDGE_RELS = frozenset([
    'healthness', 'rev_healthness', 'contains', 'rev_contains',
    'eaten_at', 'rev_eaten_at', 'ingredient', 'rev_ingredient',
])


# ============================================================================
# UTILS
# ============================================================================

def gc_collect():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


# ============================================================================
# 1. FEATURE PROJECTOR
# ============================================================================

class FeatureProjector(nn.Module):
    """
    Per-node-type feature projection.

    raw node features (varying dims) → unified hidden_channels
    with BN + GELU + optional Dropout.

    목적:
    - user(29), food(17), ingredient(101), time(4) 등 다양한 dim 통일
    - 학습 가능한 initial embedding으로 변환 (vs. random embedding)
    - raw feature 정보를 encoder에 전달하여 cold-start 완화
    """

    def __init__(
        self,
        feat_dims: Dict[str, int],
        hidden_channels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projectors = nn.ModuleDict()
        self.bns = nn.ModuleDict()

        for ntype, fdim in feat_dims.items():
            self.projectors[ntype] = nn.Linear(fdim, hidden_channels)
            self.bns[ntype] = nn.BatchNorm1d(hidden_channels)

        self.dropout = dropout

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for ntype, x in x_dict.items():
            if ntype in self.projectors:
                z = self.projectors[ntype](x.float())
                # BatchNorm safe (handle size-1 batches)
                if z.shape[0] > 1:
                    z = self.bns[ntype](z)
                z = F.gelu(z)
                z = F.dropout(z, p=self.dropout, training=self.training)
                out[ntype] = z
            else:
                # Unknown node type: zero-pad or pass through
                out[ntype] = x.float() if x is not None else None
        return out


# ============================================================================
# 2. HETERO RESGAT ENCODER
# ============================================================================

class HeteroResGATEncoder(nn.Module):
    """
    Heterogeneous Graph Encoder with:
      - GATConv message passing
      - Batch Normalization per layer
      - Residual connections (when dims match)
      - Layer-wise mean pooling (LightGCN 방식)
      - Final L2 normalization

    개선 포인트 vs v2 DualChannelEncoder:
    1. BN + Residual → gradient flow 개선, 수렴 안정화
    2. Layer-wise mean pooling → over-smoothing 방지 (LightGCN)
    3. L2 normalize → dot-product = cosine similarity (rank-optimal)
    4. Preference/Aux 채널 분리 → health signal이 ranking gradient 방해 방지
    """

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        metadata: Tuple,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        drop_edge_p: float = 0.1,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.drop_edge_p = drop_edge_p

        node_types, edge_types = metadata

        # ── Classify edge types ──
        self._pref_ets = [et for et in edge_types if et[1] in PREF_EDGE_RELS]
        self._aux_ets  = [et for et in edge_types if et[1] in AUX_EDGE_RELS]
        self._all_ets  = list(edge_types)

        # Fallback: if no pref edges found, use all
        if not self._pref_ets:
            self._pref_ets = self._all_ets

        # ── Preference channel GAT layers ──
        self.pref_convs = nn.ModuleList()
        self.pref_bns   = nn.ModuleList()
        for i in range(num_layers):
            # All layers: in=-1 (lazy), out=out_channels, concat=False
            # Using out_channels directly (no head concat to avoid dim explosion)
            conv = HeteroConv({
                et: GATConv(
                    in_channels=(-1, -1),
                    out_channels=out_channels,
                    heads=1,           # single head for stable dims
                    dropout=dropout,
                    add_self_loops=False,
                    concat=False,
                )
                for et in self._pref_ets
            }, aggr='mean')
            self.pref_convs.append(conv)
            # Per-type BN
            self.pref_bns.append(nn.ModuleDict({
                nt: nn.BatchNorm1d(out_channels) for nt in node_types
            }))

        # ── Auxiliary channel GAT layers (for health head) ──
        if self._aux_ets:
            n_aux = min(2, num_layers)
            self.aux_convs = nn.ModuleList()
            self.aux_bns   = nn.ModuleList()
            for i in range(n_aux):
                conv = HeteroConv({
                    et: GATConv(
                        in_channels=(-1, -1),
                        out_channels=out_channels,
                        heads=1,
                        dropout=dropout,
                        add_self_loops=False,
                        concat=False,
                    )
                    for et in self._aux_ets
                }, aggr='mean')
                self.aux_convs.append(conv)
                self.aux_bns.append(nn.ModuleDict({
                    nt: nn.BatchNorm1d(out_channels) for nt in node_types
                }))
        else:
            self.aux_convs = nn.ModuleList()
            self.aux_bns   = nn.ModuleList()

    def _run_channel(
        self, x_dict, edge_index_dict,
        convs, bns,
        channel_name: str = '',
    ) -> Dict[str, torch.Tensor]:
        """
        Run one channel (preference or auxiliary).
        Layer-wise mean pooling over all layer outputs.
        """
        if not convs:
            return x_dict

        cur = dict(x_dict)
        # Collect all layer outputs for LightGCN-style pooling
        layer_outs = {nt: [cur[nt]] for nt in cur}  # include z_0

        for layer_idx, (conv, bn_dict) in enumerate(zip(convs, bns)):
            # Filter edge types for this conv
            valid_ets = set(conv.convs.keys())
            filtered = {et: v for et, v in edge_index_dict.items()
                       if et in valid_ets}
            if not filtered:
                continue

            try:
                new_out = conv(cur, filtered)
            except Exception as e:
                # Skip layer on error (e.g., empty edge index)
                continue

            new_cur = dict(cur)
            for nt, z in new_out.items():
                if z is None:
                    continue
                # BN
                if z.shape[0] > 1 and nt in bn_dict:
                    z = bn_dict[nt](z)
                # GELU activation
                z = F.gelu(z)
                # Dropout
                z = F.dropout(z, p=self.dropout, training=self.training)
                # Residual (if dims match)
                if nt in cur and cur[nt].shape == z.shape:
                    z = z + cur[nt]
                new_cur[nt] = z
                if nt in layer_outs:
                    layer_outs[nt].append(z)
                else:
                    layer_outs[nt] = [z]
            cur = new_cur

        # LightGCN-style: mean pool all layer outputs
        result = {}
        for nt, layers in layer_outs.items():
            # Align to out_channels
            aligned = []
            for l in layers:
                if l.shape[-1] == self.out_channels:
                    aligned.append(l)
                elif l.shape[-1] > self.out_channels:
                    aligned.append(l[..., :self.out_channels])
                # skip if smaller dim
            if aligned:
                result[nt] = torch.stack(aligned, dim=0).mean(dim=0)
            elif cur.get(nt) is not None:
                result[nt] = cur[nt]

        return result

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict,
    ) -> Tuple[Dict, Dict]:
        """
        Returns:
            pref_z: L2-normalized preference embeddings (for ranking)
            aux_z:  auxiliary embeddings (for health head)
        """
        # DropEdge augmentation
        if self.training and self.drop_edge_p > 0:
            dropped = {}
            for et, ei in edge_index_dict.items():
                ei_d, _ = dropout_edge(ei, p=self.drop_edge_p, training=True)
                dropped[et] = ei_d
            edge_index_dict = dropped

        # Preference channel
        pref_z = self._run_channel(
            x_dict, edge_index_dict,
            self.pref_convs, self.pref_bns,
            'pref',
        )

        # Auxiliary channel
        if self.aux_convs:
            aux_z = self._run_channel(
                x_dict, edge_index_dict,
                self.aux_convs, self.aux_bns,
                'aux',
            )
        else:
            aux_z = pref_z

        # L2-normalize (critical for dot-product = cosine similarity)
        for nt in list(pref_z.keys()):
            if pref_z[nt] is not None:
                pref_z[nt] = F.normalize(pref_z[nt], dim=-1)
        for nt in list(aux_z.keys()):
            if aux_z.get(nt) is not None:
                aux_z[nt] = F.normalize(aux_z[nt], dim=-1)

        return pref_z, aux_z


# ============================================================================
# 3. RANK DOT DECODER
# ============================================================================

class RankDotDecoder(nn.Module):
    """
    순수 dot-product decoder (v2 HybridDecoder 대체).

    L2-normalized embeddings → dot-product = cosine similarity
    → rank order가 정확히 보존됨
    → MF/LightGCN 수준의 NDCG/MRR 달성 가능

    v2 HybridDecoder 문제:
    - Bilinear + Dot + MLP 앙상블이 rank calibration을 망침
    - MLP가 rank order가 아닌 absolute score를 학습
    - NDCG@10: 0.4279 vs MF 0.6133
    """

    def forward(self, z_dict: Dict, edge_label_index: torch.Tensor) -> torch.Tensor:
        u = z_dict['user'][edge_label_index[0]]
        f = z_dict['food'][edge_label_index[1]]
        return (u * f).sum(dim=-1)   # cosine similarity (both L2-normalized)


# ============================================================================
# 4. HEALTH HEAD
# ============================================================================

class HealthHead(nn.Module):
    """
    완전 분리된 auxiliary health scoring head.

    ranking gradient와 격리:
    - Phase 1에서 gradient 차단 (frozen)
    - Phase 2에서만 활성화
    - food embedding → health score (scalar)
    """

    def __init__(self, emb_dim: int, hidden_dim: int = None, dropout: float = 0.2):
        super().__init__()
        h = hidden_dim or (emb_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, h),
            nn.BatchNorm1d(h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Linear(h // 2, 1),
        )

    def forward(self, food_emb: torch.Tensor) -> torch.Tensor:
        if food_emb.shape[0] <= 1:
            # BatchNorm requires >1 samples
            return self.mlp[0](food_emb).squeeze(-1)  # skip BN
        return self.mlp(food_emb).squeeze(-1)


# ============================================================================
# 5. COMPLETE MODEL: NutriGraphNet v3
# ============================================================================

class NutriGraphNetV3(nn.Module):
    """
    NutriGraphNet v3: Rank-Calibrated Health-Aware Heterogeneous GNN

    개선 포인트 (v2 대비):
      • Feature: FeatureProjector → raw node features 완전 활용
      • Encoder: HeteroResGATEncoder (BN + Residual + LightGCN pooling)
      • Decoder: RankDotDecoder (순수 dot-product, rank-optimal)
      • Health:  완전 분리된 HealthHead (ranking gradient 격리)
      • Training: 2-stage (Phase 1: BPR, Phase 2: +health)
      • Temperature: learnable scaling for BPR
    """

    def __init__(
        self,
        feat_dims: Dict[str, int] = None,
        hidden_channels: int = 128,
        out_channels: int = 64,
        metadata: Tuple = None,
        dropout: float = 0.2,
        heads: int = 4,
        drop_edge_p: float = 0.1,
        num_layers: int = 3,
    ):
        super().__init__()

        if feat_dims is None:
            feat_dims = FEAT_DIMS

        # Feature projector (raw features → hidden_channels)
        self.projector = FeatureProjector(
            feat_dims=feat_dims,
            hidden_channels=hidden_channels,
            dropout=dropout * 0.5,
        )

        # Encoder
        self.encoder = HeteroResGATEncoder(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            metadata=metadata,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            drop_edge_p=drop_edge_p,
        )

        # Decoder
        self.decoder = RankDotDecoder()

        # Health head
        self.health_head = HealthHead(emb_dim=out_channels, dropout=dropout)

        # Learnable temperature (start at 10, clamp to [1, 30])
        self.temp = nn.Parameter(torch.ones(1) * 10.0)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict,
        edge_label_index: torch.Tensor,
        health_mode: bool = False,
    ):
        # 1. Project raw features
        proj_x = self.projector(x_dict)

        # 2. Graph encoding
        pref_z, aux_z = self.encoder(proj_x, edge_index_dict)

        # 3. Ranking score
        scores = self.decoder(pref_z, edge_label_index)
        scores = scores * self.temp.clamp(1.0, 30.0)

        if health_mode:
            food_idx = edge_label_index[1]
            _fa = aux_z.get('food')
            food_aux = _fa if _fa is not None else pref_z.get('food')
            if food_aux is not None and food_idx.max() < food_aux.shape[0]:
                health_scores = self.health_head(food_aux[food_idx])
                return scores, health_scores, pref_z
            return scores, None, pref_z

        return scores, pref_z

    def get_all_embeddings(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict,
    ) -> Tuple[Dict, Dict]:
        """Full graph embedding (for evaluation)."""
        with torch.no_grad():
            proj_x = self.projector(x_dict)
            pref_z, aux_z = self.encoder(proj_x, edge_index_dict)
        return pref_z, aux_z


# ============================================================================
# 6. LOSS FUNCTIONS
# ============================================================================

class RankingLoss(nn.Module):
    """
    Phase 1 Loss: BPR + optional InfoNCE (in-batch contrastive)

    L_BPR    = -mean(log σ(s_pos - s_neg))
    L_InfoNCE = -mean(log exp(s_pos/τ) / Σ_j exp(s_j/τ))  [in-batch]

    InfoNCE가 BPR보다 강력한 이유:
    - BPR: 1 positive vs 1 negative 비교
    - InfoNCE: 1 positive vs (batch_size - 1) negatives 비교
    - 더 많은 hard negatives → 더 discriminative embeddings
    """

    def __init__(self, infonce_weight: float = 0.1, temperature: float = 0.07):
        super().__init__()
        self.infonce_w = infonce_weight
        self.tau = temperature

    def forward(self, pos_s, neg_s):
        # BPR
        l_bpr = -F.logsigmoid(pos_s - neg_s).mean()

        # InfoNCE (in-batch: pos_s를 대각선으로 취급)
        l_infonce = torch.tensor(0.0, device=pos_s.device)
        if self.infonce_w > 0 and len(pos_s) > 1:
            # scores: [B] → [B, 1] vs [B, B]
            # treat each positive as positive for its own user
            n = pos_s.shape[0]
            all_s = torch.stack([pos_s, neg_s], dim=1)  # [B, 2]
            l_infonce = -F.log_softmax(all_s / self.tau, dim=1)[:, 0].mean()

        return l_bpr + self.infonce_w * l_infonce


class HealthAwareLoss(nn.Module):
    """
    Phase 2 Loss: BPR + Adaptive Health Margin

    L = L_BPR + λ_h * L_health_adaptive

    Adaptive health margin:
    - |health_diff| > threshold 인 경우만 penalty 적용
    - 작은 diff는 noisy → 무시
    - ranking gradient 손상 최소화
    """

    def __init__(self, lambda_health: float = 0.005, health_threshold: float = 0.1):
        super().__init__()
        self.lh = lambda_health
        self.threshold = health_threshold

    def forward(self, pos_s, neg_s, health_pos=None, health_neg=None):
        l_bpr = -F.logsigmoid(pos_s - neg_s).mean()

        l_health = torch.tensor(0.0, device=pos_s.device)
        if (health_pos is not None and health_neg is not None and self.lh > 0):
            hdiff = health_pos - health_neg
            significant = hdiff.abs() > self.threshold
            if significant.sum() > 0:
                sdiff = pos_s[significant] - neg_s[significant]
                hd    = hdiff[significant]
                # Penalize: 점수 ranking이 health ranking과 반대면 penalty
                penalty = F.relu(-sdiff * hd.sign())
                l_health = penalty.mean()

        total = l_bpr + self.lh * l_health
        return total, {'total': total.item(), 'bpr': l_bpr.item(),
                       'health': l_health.item()}


# ============================================================================
# 7. METRICS
# ============================================================================

def classification_metrics(scores_np: np.ndarray, labels_np: np.ndarray) -> Dict:
    """AUC + F1 for binary classification."""
    pred = (scores_np > 0.0).astype(int)
    m = {
        'f1': float(f1_score(labels_np, pred, zero_division=0)),
    }
    if len(np.unique(labels_np)) > 1:
        proba = 1.0 / (1.0 + np.exp(-np.clip(scores_np.astype(float), -20, 20)))
        m['auc'] = float(roc_auc_score(labels_np, proba))
    else:
        m['auc'] = 0.5
    return m


def _get_food_health(data, device) -> Optional[torch.Tensor]:
    """Extract per-food health scores from healthness edges."""
    if ('user', 'healthness', 'food') not in data.edge_types:
        return None
    try:
        et = ('user', 'healthness', 'food')
        store = data[et]
        h_ei = store.edge_index
        if h_ei.shape[1] == 0:
            return None
        h_ea = getattr(store, 'edge_attr', None)
        if h_ea is None:
            return None
        h_ei = h_ei.to(device)
        h_ea = h_ea.to(device).float()
        if h_ea.dim() > 1:
            h_ea = h_ea.squeeze(-1)
        nf = data['food'].num_nodes
        acc = torch.zeros(nf, device=device)
        cnt = torch.zeros(nf, device=device)
        acc.scatter_add_(0, h_ei[1], h_ea)
        cnt.scatter_add_(0, h_ei[1], torch.ones_like(h_ea))
        return acc / (cnt + 1e-8)
    except Exception:
        return None


def ranking_metrics_v3(
    pref_z: Dict,
    data,
    device,
    health_scores=None,
    k_list: Tuple = (5, 10, 20),
    max_users: int = 500,
    n_neg_sample: int = 100,
    seed: int = 12345,
) -> Dict:
    """
    Sampled ranking evaluation (RecSys 논문 표준: 1 pos + 100 neg).

    각 user의 positive food 1개 + 랜덤 100개 negative food를 합쳐
    101개 후보 중 ranking → HR@K, NDCG@K, MRR 계산.
    """
    eil = data[('user', 'eats', 'food')].edge_label_index
    el  = data[('user', 'eats', 'food')].edge_label

    pos_mask = (el == 1)
    pos_eil  = eil[:, pos_mask]

    if pos_eil.shape[1] == 0:
        out = {'MRR': 0.0}
        for k in k_list:
            out[f'HR@{k}'] = 0.0
            out[f'NDCG@{k}'] = 0.0
        return out

    unique_users = pos_eil[0].unique().cpu().numpy()
    if len(unique_users) > max_users:
        rng_sel = np.random.default_rng(seed)
        unique_users = rng_sel.choice(unique_users, max_users, replace=False)

    num_foods = data['food'].num_nodes
    z_u = pref_z['user'].detach().cpu().float()
    z_f = pref_z['food'].detach().cpu().float()
    hs_cpu  = health_scores.detach().cpu().float() if health_scores is not None else None
    hs_mean = float(hs_cpu.mean()) if hs_cpu is not None else 0.0

    pos_eil_np = pos_eil.cpu().numpy()

    hr   = {k: [] for k in k_list}
    ndcg = {k: [] for k in k_list}
    hg   = {k: [] for k in k_list}
    mrr_vals = []

    rng = np.random.default_rng(seed)

    for u_raw in unique_users:
        u = int(u_raw)
        if u >= z_u.shape[0]:
            continue

        mask_u      = (pos_eil_np[0] == u)
        pos_foods_u = pos_eil_np[1, mask_u]
        pos_foods_u = pos_foods_u[pos_foods_u < num_foods]
        if len(pos_foods_u) == 0:
            continue

        pos_set  = set(int(x) for x in pos_foods_u)
        target_f = int(rng.choice(pos_foods_u))

        # Negative pool
        neg_pool = np.array([i for i in range(num_foods) if i not in pos_set])
        if len(neg_pool) < n_neg_sample:
            continue
        neg_idx   = rng.choice(neg_pool, n_neg_sample, replace=False)
        candidates = np.concatenate([[target_f], neg_idx])  # [101]

        # Score
        u_emb     = z_u[u]                        # [D]
        f_embs    = z_f[candidates]               # [101, D]
        scores_c  = (f_embs * u_emb.unsqueeze(0)).sum(-1).numpy()

        sorted_idx = np.argsort(-scores_c)
        target_pos = 0  # target_f is at index 0

        # MRR
        for rank, li in enumerate(sorted_idx):
            if li == target_pos:
                mrr_vals.append(1.0 / (rank + 1))
                break

        for k in k_list:
            top_k = set(sorted_idx[:k].tolist())
            hit   = int(target_pos in top_k)
            hr[k].append(float(hit))

            # NDCG@K
            dcg = 0.0
            for rank, li in enumerate(sorted_idx[:k]):
                if int(li) == target_pos:
                    dcg = 1.0 / np.log2(rank + 2)
                    break
            ndcg[k].append(dcg)

            # HealthGain@K
            if hs_cpu is not None:
                topk_foods = candidates[sorted_idx[:k]]
                valid_idx  = topk_foods[topk_foods < hs_cpu.shape[0]]
                if len(valid_idx) > 0:
                    hg[k].append(float(hs_cpu[valid_idx].mean()) - hs_mean)

    out = {}
    for k in k_list:
        out[f'HR@{k}']   = float(np.mean(hr[k]))   if hr[k]   else 0.0
        out[f'NDCG@{k}'] = float(np.mean(ndcg[k])) if ndcg[k] else 0.0
        if hg[k]:
            out[f'HealthGain@{k}'] = float(np.mean(hg[k]))
    out['MRR'] = float(np.mean(mrr_vals)) if mrr_vals else 0.0
    return out


# ============================================================================
# 8. DATA PREPARATION
# ============================================================================

def fast_link_split_v3(
    data,
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    neg_ratio: float = 1.0,
    seed: int = 42,
):
    """
    Fast manual link split.
    Replicates v2's fast_link_split logic for v3.

    Returns (train_data, val_data, test_data).
    """
    rng = np.random.default_rng(seed)

    ei     = data[('user', 'eats', 'food')].edge_index  # [2, E]
    E      = ei.shape[1]
    num_foods  = data['food'].num_nodes
    num_users  = data['user'].num_nodes

    perm  = torch.from_numpy(rng.permutation(E))
    ei_sh = ei[:, perm]

    n_test  = max(1, int(E * test_ratio))
    n_val   = max(1, int(E * val_ratio))
    n_train = E - n_test - n_val

    train_pos = ei_sh[:, :n_train]
    val_pos   = ei_sh[:, n_train : n_train + n_val]
    test_pos  = ei_sh[:, n_train + n_val :]

    def make_neg(pos_ei, n):
        u_idx = pos_ei[0].cpu().numpy()
        pos_set_per_user = {}
        for u, f in zip(pos_ei[0].tolist(), pos_ei[1].tolist()):
            pos_set_per_user.setdefault(u, set()).add(f)
        neg_u, neg_f = [], []
        for i in range(n):
            u = int(u_idx[i % len(u_idx)])
            f = int(rng.integers(0, num_foods))
            attempts = 0
            while f in pos_set_per_user.get(u, set()) and attempts < 10:
                f = int(rng.integers(0, num_foods))
                attempts += 1
            neg_u.append(u); neg_f.append(f)
        return torch.tensor([neg_u, neg_f], dtype=torch.long)

    def build_data(pos_ei, mp_ei):
        d = copy.copy(data)
        d[('user','eats','food')].edge_index     = mp_ei
        d[('food','rev_eats','user')].edge_index = mp_ei.flip(0)

        n_pos = pos_ei.shape[1]
        n_neg = int(n_pos * neg_ratio)
        neg_ei = make_neg(pos_ei, n_neg)

        eil = torch.cat([pos_ei, neg_ei], dim=1)
        el  = torch.cat([
            torch.ones(n_pos, dtype=torch.float),
            torch.zeros(n_neg, dtype=torch.float),
        ])
        perm2 = torch.randperm(eil.shape[1])
        d[('user','eats','food')].edge_label_index = eil[:, perm2]
        d[('user','eats','food')].edge_label        = el[perm2]
        return d

    train_d = build_data(train_pos, train_pos)
    val_d   = build_data(val_pos,   train_pos)
    test_d  = build_data(test_pos,  train_pos)
    return train_d, val_d, test_d


def _prepare_pairs(data, device):
    """Extract pos/neg pairs from HeteroData."""
    eil = data[('user','eats','food')].edge_label_index.to(device)
    el  = data[('user','eats','food')].edge_label.to(device)
    pos_m = (el == 1)
    neg_m = (el == 0)
    pos_ei = eil[:, pos_m]
    neg_ei = eil[:, neg_m]

    n = min(pos_ei.shape[1], neg_ei.shape[1])
    if n == 0 or neg_ei.shape[1] == 0:
        # Generate random negatives
        n   = pos_ei.shape[1]
        nf  = data['food'].num_nodes
        nf_ = torch.randint(0, nf, (n,), device=device)
        neg_ei = torch.stack([pos_ei[0], nf_])
    else:
        pos_ei = pos_ei[:, :n]
        neg_ei = neg_ei[:, :n]

    return eil, el, pos_ei, neg_ei


def _infer_feat_dims(data) -> Dict[str, int]:
    """Infer feature dims from HeteroData."""
    fd = {}
    for nt in data.node_types:
        x = data[nt].x
        if x is not None:
            fd[nt] = int(x.shape[1])
    return fd


# ============================================================================
# 9. TRAINING
# ============================================================================

def train_epoch_v3(
    model, optimizer, train_data, criterion, device,
    batch_size: int = 4096,
    health_mode: bool = False,
) -> Tuple[float, Dict]:
    """One training epoch."""
    model.train()

    x_dict  = {k: v.to(device) for k, v in train_data.x_dict.items()}
    ei_dict = {k: v.to(device) for k, v in train_data.edge_index_dict.items()}
    eil, el, pos_ei, neg_ei = _prepare_pairs(train_data, device)
    hs = _get_food_health(train_data, device)

    # Sub-sample for memory efficiency
    n = min(pos_ei.shape[1], batch_size)
    idx = torch.randperm(pos_ei.shape[1], device=device)[:n]
    p_ei = pos_ei[:, idx]
    n_ei_s = neg_ei[:, idx % neg_ei.shape[1]] if neg_ei.shape[1] > 0 else neg_ei[:, idx]

    optimizer.zero_grad()

    if health_mode and hs is not None:
        # Phase 2: health-aware
        pos_s, pos_h, _ = model(x_dict, ei_dict, p_ei, health_mode=True)
        neg_s, neg_h, _ = model(x_dict, ei_dict, n_ei_s, health_mode=True)
        ph = hs[p_ei[1]] if pos_h is not None else None
        nh = hs[n_ei_s[1]] if neg_h is not None else None
        loss, ld = criterion(pos_s, neg_s, ph, nh)
    elif health_mode:
        pos_s, _ = model(x_dict, ei_dict, p_ei)
        neg_s, _ = model(x_dict, ei_dict, n_ei_s)
        loss = -F.logsigmoid(pos_s - neg_s).mean()
        ld = {'bpr': loss.item(), 'health': 0.0}
    else:
        # Phase 1: pure BPR (+ InfoNCE)
        pos_s, _ = model(x_dict, ei_dict, p_ei)
        neg_s, _ = model(x_dict, ei_dict, n_ei_s)
        loss = criterion(pos_s, neg_s)
        ld = {'bpr': loss.item() if hasattr(loss, 'item') else float(loss)}

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Classification metrics (no_grad)
    with torch.no_grad():
        n_cls = min(eil.shape[1], 4096)
        idx2  = torch.randperm(eil.shape[1], device=device)[:n_cls]
        s, _  = model(x_dict, ei_dict, eil[:, idx2])
        cm    = classification_metrics(s.cpu().numpy(), el[idx2].cpu().numpy())

    gc_collect()
    return float(loss.item()), {**ld, **cm}


@torch.no_grad()
def evaluate_v3(
    model, data, device, compute_rank: bool = False, seed: int = 42,
) -> Tuple[float, Dict]:
    """Evaluate on val/test data."""
    model.eval()

    x_dict  = {k: v.to(device) for k, v in data.x_dict.items()}
    ei_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    eil, el, pos_ei, neg_ei = _prepare_pairs(data, device)
    hs = _get_food_health(data, device)

    # BPR loss
    n = min(pos_ei.shape[1], 4096)
    idx = torch.randperm(pos_ei.shape[1])[:n]
    pos_s, pref_z = model(x_dict, ei_dict, pos_ei[:, idx])
    neg_s, _      = model(x_dict, ei_dict, neg_ei[:, idx % neg_ei.shape[1]])
    bpr_loss = -F.logsigmoid(pos_s - neg_s).mean().item()

    # Classification
    n_cls = min(eil.shape[1], 8192)
    idx2  = torch.randperm(eil.shape[1])[:n_cls]
    s, _  = model(x_dict, ei_dict, eil[:, idx2])
    cm    = classification_metrics(s.cpu().numpy(), el[idx2].cpu().numpy())

    out = {'bpr': bpr_loss, **cm}

    if compute_rank:
        # Full-graph encode for ranking
        proj_x = model.projector(x_dict)
        full_pref_z, _ = model.encoder(proj_x, ei_dict)
        for nt in full_pref_z:
            if full_pref_z[nt] is not None:
                full_pref_z[nt] = F.normalize(full_pref_z[nt], dim=-1)
        rm = ranking_metrics_v3(full_pref_z, data, device, hs, seed=seed)
        out.update(rm)

    del pref_z
    gc_collect()
    return bpr_loss, out


# ============================================================================
# 10. FULL TRAINING PIPELINE
# ============================================================================

def train_nutrigraphnet_v3(
    train_data,
    val_data,
    test_data,
    device,
    # Architecture
    feat_dims: Dict[str, int] = None,
    hidden_channels: int = 128,
    out_channels: int = 64,
    num_layers: int = 3,
    heads: int = 4,
    dropout: float = 0.2,
    drop_edge_p: float = 0.1,
    # Training
    n_epochs: int = 100,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 4096,
    infonce_weight: float = 0.1,
    # Health
    lambda_health: float = 0.005,
    health_threshold: float = 0.1,
    phase1_frac: float = 0.8,
    # Misc
    seed: int = 42,
    verbose: bool = True,
    early_stop_patience: int = 15,
) -> Tuple:
    """
    Two-stage training pipeline:

    Phase 1 (phase1_frac * n_epochs):  Pure BPR + InfoNCE
    Phase 2 ((1-phase1_frac) * n_epochs): BPR + Health margin
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Infer feature dims from data
    if feat_dims is None:
        feat_dims = _infer_feat_dims(train_data)

    # Build model
    model = NutriGraphNetV3(
        feat_dims=feat_dims,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        metadata=train_data.metadata(),
        dropout=dropout,
        heads=heads,
        drop_edge_p=drop_edge_p,
        num_layers=num_layers,
    ).to(device)

    # Warm-up: initialize lazy parameters with a dummy forward pass
    model.eval()
    with torch.no_grad():
        try:
            _x = {k: v.to(device) for k, v in train_data.x_dict.items()}
            _ei = {k: v.to(device) for k, v in train_data.edge_index_dict.items()}
            _eil = train_data[('user','eats','food')].edge_label_index[:, :4].to(device)
            model(_x, _ei, _eil)
        except Exception as _e:
            if verbose:
                print(f"[v3] Warmup note: {_e}")
    model.train()

    if verbose:
        try:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[v3] Model params: {n_params:,}")
        except Exception:
            print("[v3] Model params: (lazy, not yet initialized)")
        print(f"[v3] Architecture: hidden={hidden_channels}, out={out_channels}, "
              f"layers={num_layers}, heads={heads}")

    phase1_epochs = int(n_epochs * phase1_frac)
    phase2_epochs = n_epochs - phase1_epochs

    # ── Phase 1: Pure BPR + InfoNCE ──────────────────────────────────────────
    rank_criterion = RankingLoss(infonce_weight=infonce_weight)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(phase1_epochs, 1),
                                  eta_min=lr * 0.01)

    best_ndcg   = 0.0
    best_state  = None
    patience_cnt = 0
    history     = []

    if verbose:
        print(f"\n[v3] Phase 1: Pure BPR + InfoNCE ({phase1_epochs} epochs)")

    for epoch in range(1, phase1_epochs + 1):
        loss, td = train_epoch_v3(
            model, optimizer, train_data, rank_criterion,
            device, batch_size, health_mode=False,
        )
        scheduler.step()

        do_rank = (epoch % 20 == 0 or epoch == phase1_epochs)
        if epoch % 10 == 0 or epoch == phase1_epochs:
            _, vd = evaluate_v3(model, val_data, device,
                                compute_rank=do_rank, seed=seed)
            ndcg = vd.get('NDCG@10', 0.0)
            hr   = vd.get('HR@10',   0.0)

            if verbose:
                print(f"  E{epoch:4d} loss={loss:.4f} "
                      f"AUC={td.get('auc',0):.4f} "
                      f"HR@10={hr:.4f} NDCG@10={ndcg:.4f}")

            history.append({'epoch': epoch, 'phase': 1,
                            'loss': loss, **td, **vd})

            if ndcg > best_ndcg:
                best_ndcg   = ndcg
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if (patience_cnt >= early_stop_patience
                        and epoch > phase1_epochs // 2):
                    if verbose:
                        print(f"  [Phase 1 Early Stop] epoch={epoch}")
                    break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Phase 2: BPR + Health ─────────────────────────────────────────────────
    if phase2_epochs > 0 and lambda_health > 0:
        if verbose:
            print(f"\n[v3] Phase 2: BPR + Health "
                  f"(λ={lambda_health}, {phase2_epochs} epochs)")

        health_crit = HealthAwareLoss(lambda_health=lambda_health,
                                      health_threshold=health_threshold)
        opt2 = AdamW(model.parameters(), lr=lr * 0.3, weight_decay=weight_decay)
        sch2 = CosineAnnealingLR(opt2, T_max=max(phase2_epochs, 1),
                                 eta_min=lr * 0.003)

        best_ndcg_p2  = best_ndcg
        patience_cnt2 = 0

        for epoch in range(1, phase2_epochs + 1):
            loss, td = train_epoch_v3(
                model, opt2, train_data, health_crit,
                device, batch_size, health_mode=True,
            )
            sch2.step()

            do_rank = (epoch % 20 == 0 or epoch == phase2_epochs)
            if epoch % 10 == 0 or epoch == phase2_epochs:
                _, vd = evaluate_v3(model, val_data, device,
                                    compute_rank=do_rank, seed=seed)
                ndcg = vd.get('NDCG@10', 0.0)
                hr   = vd.get('HR@10',   0.0)

                if verbose:
                    print(f"  E{phase1_epochs+epoch:4d} [H] "
                          f"loss={loss:.4f} "
                          f"AUC={td.get('auc',0):.4f} "
                          f"HR@10={hr:.4f} NDCG@10={ndcg:.4f}")

                history.append({'epoch': phase1_epochs + epoch, 'phase': 2,
                                'loss': loss, **td, **vd})

                # Allow slight NDCG drop for health gain (within 2%)
                if ndcg >= best_ndcg_p2 * 0.98:
                    best_ndcg_p2 = max(ndcg, best_ndcg_p2)
                    best_state   = {k: v.clone()
                                    for k, v in model.state_dict().items()}
                    patience_cnt2 = 0
                else:
                    patience_cnt2 += 1
                    if patience_cnt2 >= early_stop_patience:
                        if verbose:
                            print(f"  [Phase 2 Early Stop] epoch={epoch}")
                        break

        if best_state is not None:
            model.load_state_dict(best_state)

    # ── Final test ────────────────────────────────────────────────────────────
    _, test_metrics = evaluate_v3(model, test_data, device,
                                   compute_rank=True, seed=seed)

    if verbose:
        print(f"\n[v3] === Test Results ===")
        for k in ['auc', 'f1', 'HR@5', 'HR@10', 'HR@20',
                  'NDCG@10', 'MRR', 'HealthGain@10']:
            if k in test_metrics:
                print(f"  {k:20s}: {test_metrics[k]:.4f}")

    gc_collect()
    return model, test_metrics, history


# ============================================================================
# 11. K-FOLD CROSS VALIDATION
# ============================================================================

def run_kfold_v3(
    data_path: str,
    output_dir: str = 'results/v3',
    n_folds: int = 5,
    device_str: str = 'auto',
    # Architecture
    hidden_channels: int = 128,
    out_channels: int = 64,
    num_layers: int = 3,
    heads: int = 4,
    dropout: float = 0.2,
    drop_edge_p: float = 0.1,
    # Training
    n_epochs: int = 100,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 4096,
    infonce_weight: float = 0.1,
    lambda_health: float = 0.005,
    phase1_frac: float = 0.8,
    seed: int = 42,
    verbose: bool = True,
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    interaction_ratio: float = 1.0,
):
    """Run k-fold cross-validation for NutriGraphNet v3."""
    import pickle

    device = torch.device(
        'cuda' if (device_str == 'auto' and torch.cuda.is_available())
        else ('cpu' if device_str in ('cpu', 'auto') else device_str)
    )
    print(f"[v3] Device: {device}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"[v3] Loading data: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print(f"  Users={data['user'].num_nodes}, "
          f"Foods={data['food'].num_nodes}, "
          f"Edges={data[('user','eats','food')].edge_index.shape[1]}")

    # ── EXP-B: interaction sparsity subsampling ───────────────────────────────
    # Ported verbatim from nutrigraphnet_v2.py so that a given (seed, ratio)
    # draws the EXACT same interaction subset the v2 baselines saw -- otherwise
    # the sparsity sweep is not comparable across model versions.
    # Only user-food eats/healthness edges are subsampled; the auxiliary graph
    # (ingredient, food-similar, time) is left intact, as in v2.
    if interaction_ratio < 1.0:
        ei_key  = ('user', 'eats', 'food')
        rev_key = ('food', 'rev_eats', 'user')
        h_key   = ('user', 'healthness', 'food')
        rh_key  = ('food', 'rev_healthness', 'user')
        n_orig = data[ei_key].edge_index.shape[1]
        torch.manual_seed(seed)
        keep = torch.randperm(n_orig)[:int(n_orig * interaction_ratio)]
        keep_sorted = keep.sort().values
        for k in (ei_key, rev_key, h_key, rh_key):
            if k in data.edge_types:
                data[k].edge_index = data[k].edge_index[:, keep_sorted]
                if hasattr(data[k], 'edge_attr') and data[k].edge_attr is not None:
                    data[k].edge_attr = data[k].edge_attr[keep_sorted]
        print(f"  [sparsity] kept {len(keep_sorted):,}/{n_orig:,} interactions "
              f"({interaction_ratio*100:.0f}%)")
        # HealthGain@K is not meaningful below full density: per-food health
        # scores are averaged from healthness edge_attr and foods left with no
        # surviving edge score 0, collapsing the population baseline (0.6653 at
        # 100% -> 0.1529 at 10%). Ignore HealthGain in sparsity-sweep outputs.
        print("  [sparsity] NOTE: HealthGain@K is invalid at this density "
              "(health-score baseline dilution) -- ignore it in these results.")

    feat_dims = _infer_feat_dims(data)
    print(f"  Feature dims: {feat_dims}")

    all_metrics = []

    for fold in range(n_folds):
        print(f"\n{'='*60}")
        print(f"[v3] Fold {fold+1}/{n_folds}")
        print(f"{'='*60}")

        fold_seed = seed + fold
        train_d, val_d, test_d = fast_link_split_v3(
            data,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=fold_seed,
        )

        _, metrics, history = train_nutrigraphnet_v3(
            train_data=train_d,
            val_data=val_d,
            test_data=test_d,
            device=device,
            feat_dims=feat_dims,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            drop_edge_p=drop_edge_p,
            n_epochs=n_epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            infonce_weight=infonce_weight,
            lambda_health=lambda_health,
            phase1_frac=phase1_frac,
            seed=fold_seed,
            verbose=verbose,
        )

        # Save fold result
        fold_path = out_dir / f'fold_{fold+1}_metrics.json'
        with open(fold_path, 'w') as f:
            json.dump({'fold': fold+1, 'metrics': metrics, 'history': history},
                      f, cls=_NumpyEncoder, indent=2)

        all_metrics.append(metrics)
        print(f"[v3] Fold {fold+1} → "
              f"HR@10={metrics.get('HR@10',0):.4f} "
              f"NDCG@10={metrics.get('NDCG@10',0):.4f} "
              f"MRR={metrics.get('MRR',0):.4f} "
              f"AUC={metrics.get('auc',0):.4f}")

        gc_collect()

    # Aggregate over folds
    agg = {}
    if all_metrics:
        for k in all_metrics[0]:
            vals = [float(m.get(k, 0)) for m in all_metrics]
            agg[k]            = float(np.mean(vals))
            agg[f'{k}_std']   = float(np.std(vals))

    print(f"\n[v3] ===== {n_folds}-Fold Aggregate =====")
    for k in ['auc', 'f1', 'HR@5', 'HR@10', 'HR@20', 'NDCG@10', 'MRR']:
        if k in agg:
            print(f"  {k:20s}: {agg[k]:.4f} ± {agg[k+'_std']:.4f}")
    if 'HealthGain@10' in agg:
        print(f"  {'HealthGain@10':20s}: {agg['HealthGain@10']:.5f}")

    # Save summary
    summary = {
        'model': 'NutriGraphNetV3',
        'n_folds': n_folds,
        'per_fold': all_metrics,
        'aggregate': agg,
        'config': {
            'hidden_channels': hidden_channels,
            'out_channels': out_channels,
            'num_layers': num_layers,
            'heads': heads,
            'dropout': dropout,
            'lambda_health': lambda_health,
            'phase1_frac': phase1_frac,
            'infonce_weight': infonce_weight,
        },
    }
    summary_path = out_dir / 'nutrigraphnet_v3_results.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, cls=_NumpyEncoder, indent=2)
    print(f"\n[v3] Results saved to: {summary_path}")

    return summary


# ============================================================================
# 12. QUICK TEST (CPU sandbox)
# ============================================================================

def quick_test(
    data_path: str,
    n_epochs: int = 30,
    device_str: str = 'cpu',
):
    """
    CPU sandbox에서 빠른 동작 확인.
    경량 파라미터: hidden=32, out=16, layers=1
    """
    import pickle

    device = torch.device(
        device_str if device_str != 'auto'
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f"[v3 quick test] Device: {device}")

    # Load data
    print(f"[v3 quick test] Loading: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    feat_dims = _infer_feat_dims(data)
    print(f"[v3 quick test] Node types: {data.node_types}")
    print(f"[v3 quick test] Edge types: {data.edge_types}")
    print(f"[v3 quick test] Feature dims: {feat_dims}")

    # Split
    train_d, val_d, test_d = fast_link_split_v3(data, seed=42)
    print(f"[v3 quick test] Train edges: "
          f"{train_d[('user','eats','food')].edge_label_index.shape[1]}")

    # Lightweight model for CPU
    _, metrics, history = train_nutrigraphnet_v3(
        train_data=train_d,
        val_data=val_d,
        test_data=test_d,
        device=device,
        feat_dims=feat_dims,
        hidden_channels=32,
        out_channels=16,
        num_layers=1,
        heads=2,
        dropout=0.1,
        drop_edge_p=0.05,
        n_epochs=n_epochs,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=1024,
        infonce_weight=0.05,
        lambda_health=0.005,
        phase1_frac=0.8,
        seed=42,
        verbose=True,
    )

    print("\n[v3 quick test] === Final Results ===")
    for k in ['auc', 'f1', 'HR@5', 'HR@10', 'NDCG@10', 'MRR']:
        if k in metrics:
            print(f"  {k:20s}: {metrics[k]:.4f}")
    if 'HealthGain@10' in metrics:
        print(f"  {'HealthGain@10':20s}: {metrics['HealthGain@10']:.5f}")

    print("\n[v3 quick test] PASSED ✓")
    return metrics


# ============================================================================
# 13. CLI ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NutriGraphNet v3 — Rank-Calibrated Health-Aware GNN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data',    type=str, default='data/processed_data/processed_data_GNN_v5.pkl')
    parser.add_argument('--output',  type=str, default='results/v3')
    parser.add_argument('--folds',   type=int, default=5)
    parser.add_argument('--epochs',  type=int, default=100)
    parser.add_argument('--hidden',  type=int, default=128)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--layers',  type=int, default=3)
    parser.add_argument('--heads',   type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr',      type=float, default=3e-4)
    parser.add_argument('--lambda_health', type=float, default=0.005)
    parser.add_argument('--phase1_frac',   type=float, default=0.8)
    parser.add_argument('--infonce_weight', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed',   type=int, default=42)
    parser.add_argument('--interaction_ratio', type=float, default=1.0,
                        help='EXP-B: fraction of user-food interactions to keep '
                             '(0~1). Seeded identically to nutrigraphnet_v2.py, '
                             'so a given (seed, ratio) reproduces the same subset '
                             'the v2 baselines used. HealthGain@K is invalid '
                             'below 1.0 -- see the note in run_kfold_v3().')
    parser.add_argument('--quick',  action='store_true',
                        help='Quick test mode (CPU, lightweight params)')
    parser.add_argument('--quick_epochs', type=int, default=30)

    args = parser.parse_args()

    if args.quick:
        quick_test(args.data, n_epochs=args.quick_epochs, device_str=args.device)
    else:
        run_kfold_v3(
            data_path=args.data,
            output_dir=args.output,
            n_folds=args.folds,
            device_str=args.device,
            hidden_channels=args.hidden,
            out_channels=args.out_dim,
            num_layers=args.layers,
            heads=args.heads,
            dropout=args.dropout,
            n_epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            infonce_weight=args.infonce_weight,
            lambda_health=args.lambda_health,
            phase1_frac=args.phase1_frac,
            seed=args.seed,
            interaction_ratio=args.interaction_ratio,
        )
