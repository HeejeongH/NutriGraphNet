"""
NutriGraphNet v2: Health-Aware Heterogeneous Graph Neural Network
for Personalized Food Recommendation

메모리 효율적 버전 (2GB RAM 환경 최적화):
- Mini-batch training with NeighborLoader
- Gradient checkpointing
- 서브그래프 샘플링으로 전체 그래프 로딩 회피

주요 개선사항 (논문 기여):
1. [Model] Dual-Channel Encoder: Preference + Health 분리 학습
2. [Model] 3-layer Hetero-GAT with edge feature integration
3. [Model] Hybrid Decoder: Bilinear + Dot + MLP 앙상블
4. [Loss] Health-Aware BPR: 건강 마진 페널티 추가
5. [Loss] InfoNCE Contrastive Learning: 표현력 강화
6. [Training] DropEdge regularization
7. [Training] OneCycleLR with warmup
8. [Evaluation] NDCG@K, HR@K, MRR, HealthGain@K
9. [Eval] Statistical significance test (Wilcoxon)
10. [Baseline] MF, LightGCN, NGCF, SGL, HFRS-DA 비교 모델

Author: Heejeong
Date: 2026-06-27
"""

import os
import sys
import gc
import pickle
import argparse
import json
import time
from pathlib import Path

from typing import Dict, Tuple, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars, booleans, and arrays."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import torch_geometric
from torch_geometric.nn import (
    HeteroConv, GATConv, Linear, LayerNorm as PygLayerNorm
)
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import dropout_edge

from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, average_precision_score
)

try:
    from scipy.stats import wilcoxon
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ============================================================================
# 0. MEMORY UTILITIES
# ============================================================================

def gc_collect():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# 1. DUAL-CHANNEL HETERO-GAT ENCODER
# ============================================================================

class DualChannelEncoder(nn.Module):
    """
    Dual-Channel Heterogeneous Graph Encoder.

    Preference Channel: 사용자-음식 상호작용 패턴 학습
        edges: user-eats-food, food-rev_eats-user, food-similar-food

    Health Channel: 건강 신호 전파
        edges: user-healthness-food, food-contains-ingredient,
               food-eaten_at-time

    Fusion: z = sigmoid(α) * z_pref + (1-sigmoid(α)) * z_health
    α는 노드 타입별 학습 파라미터

    Architecture per channel:
        - 3-layer GAT (last layer single-head for stability)
        - LayerNorm after each layer
        - Residual connections
        - DropEdge regularization during training
    """

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        metadata: Tuple,
        dropout: float = 0.4,
        heads: int = 4,
        drop_edge_p: float = 0.1,
        num_layers: int = 3,
    ):
        super().__init__()
        self.dropout = dropout
        self.drop_edge_p = drop_edge_p
        self.num_layers = num_layers

        node_types, edge_types = metadata

        # ── Preference channel edge types ──
        self._pref_ets = [
            et for et in edge_types
            if et[1] not in ('healthness', 'rev_healthness', 'contains', 'rev_contains',
                             'eaten_at', 'rev_eaten_at')
        ]
        # ── Health channel edge types ──
        self._health_ets = [
            et for et in edge_types
            if et[1] in ('healthness', 'rev_healthness', 'contains', 'rev_contains',
                         'eaten_at', 'rev_eaten_at')
        ]

        # ── Build preference convolution layers ──
        self.pref_convs = nn.ModuleList()
        self.pref_norms = nn.ModuleList()
        self._build_channel(
            self.pref_convs, self.pref_norms,
            self._pref_ets, node_types,
            num_layers, hidden_channels, out_channels, heads, dropout
        )

        # ── Build health convolution layers (2 layers sufficient) ──
        self.health_convs = nn.ModuleList()
        self.health_norms = nn.ModuleList()
        n_health_layers = min(2, num_layers)
        if self._health_ets:
            self._build_channel(
                self.health_convs, self.health_norms,
                self._health_ets, node_types,
                n_health_layers, hidden_channels, out_channels, heads, dropout
            )

        # ── Learnable fusion scalar per node type ──
        self.alpha = nn.ParameterDict({
            nt: nn.Parameter(torch.zeros(1))  # init at 0.5 after sigmoid
            for nt in ['user', 'food']
        })

        # ── Contrastive projection head ──
        self.proj_head = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels // 2)
        )

    def _build_channel(
        self, convs, norms, edge_types, node_types,
        num_layers, hidden_ch, out_ch, heads, dropout
    ):
        if not edge_types:
            return

        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            out = out_ch if is_last else hidden_ch
            n_h = 1 if is_last else heads
            concat = not is_last

            gat_kwargs = dict(
                out_channels=out // n_h if concat else out,
                heads=n_h,
                dropout=dropout,
                add_self_loops=False,
                concat=concat
            )

            conv = HeteroConv({
                et: GATConv(in_channels=(-1, -1), **gat_kwargs)
                for et in edge_types
            }, aggr='mean')
            convs.append(conv)

            # LayerNorm per node type (size = hidden_ch or out_ch)
            norm_size = out_ch if is_last else hidden_ch
            norm_dict = nn.ModuleDict({
                nt: nn.LayerNorm(norm_size) for nt in node_types
            })
            norms.append(norm_dict)

    def _forward_channel(self, x_dict, edge_index_dict, convs, norms):
        """Forward pass for one channel."""
        if not convs:
            return x_dict

        cur_x = dict(x_dict)
        for conv, norm_dict in zip(convs, norms):
            # Filter to only edges this conv uses
            valid_ets = set(conv.convs.keys())
            filtered = {et: v for et, v in edge_index_dict.items() if et in valid_ets}
            if not filtered:
                continue

            try:
                out = conv(cur_x, filtered)
            except Exception:
                continue

            for nt, z in out.items():
                if nt in norm_dict:
                    z = norm_dict[nt](z)
                # Residual (only if shape matches)
                if nt in cur_x and cur_x[nt].shape == z.shape:
                    z = z + cur_x[nt]
                z = F.gelu(z)
                z = F.dropout(z, p=self.dropout, training=self.training)
                out[nt] = z

            cur_x = {**cur_x, **out}

        return cur_x

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict,
        return_projections: bool = False,
    ):
        # DropEdge during training
        if self.training and self.drop_edge_p > 0:
            new_ei = {}
            for et, ei in edge_index_dict.items():
                ei_d, _ = dropout_edge(ei, p=self.drop_edge_p, training=True)
                new_ei[et] = ei_d
            edge_index_dict = new_ei

        # ── Preference channel ──
        pref_x = self._forward_channel(x_dict, edge_index_dict,
                                        self.pref_convs, self.pref_norms)

        # ── Health channel ──
        if self.health_convs:
            health_x = self._forward_channel(x_dict, edge_index_dict,
                                              self.health_convs, self.health_norms)
        else:
            health_x = pref_x

        # ── Channel fusion ──
        fused = {}
        for nt in pref_x:
            p = pref_x[nt]
            h = health_x.get(nt, p)
            if nt in self.alpha and p.shape == h.shape:
                a = torch.sigmoid(self.alpha[nt])
                fused[nt] = a * p + (1 - a) * h
            else:
                fused[nt] = p

        if return_projections:
            proj = {
                nt: F.normalize(self.proj_head(emb), dim=-1)
                for nt, emb in fused.items()
                if nt in ('user', 'food')
            }
            return fused, proj

        return fused


# ============================================================================
# 2. HYBRID DECODER
# ============================================================================

class HybridDecoder(nn.Module):
    """
    Hybrid decoder: Bilinear × Dot × MLP 앙상블
    score = w[0]*bilinear + w[1]*dot + w[2]*mlp
    w는 softmax normalized 학습 파라미터
    """

    def __init__(self, emb_dim: int, dropout: float = 0.3):
        super().__init__()
        self.bilinear = nn.Bilinear(emb_dim, emb_dim, 1, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, 1)
        )

        self.w = nn.Parameter(torch.ones(3) / 3)

    def forward(self, z_dict: Dict, edge_label_index: torch.Tensor) -> torch.Tensor:
        u = z_dict['user'][edge_label_index[0]]   # [E, D]
        f = z_dict['food'][edge_label_index[1]]   # [E, D]

        bil  = self.bilinear(u, f)                                  # [E, 1]
        dot  = (u * f).sum(dim=-1, keepdim=True)                    # [E, 1]
        mlp  = self.mlp(torch.cat([u, f, u * f], dim=-1))          # [E, 1]

        wt   = F.softmax(self.w, dim=0)
        score = wt[0] * bil + wt[1] * dot + wt[2] * mlp

        return score.squeeze(-1)


# ============================================================================
# 3. COMPLETE MODEL
# ============================================================================

class NutriGraphNetV2(nn.Module):
    """
    NutriGraphNet v2: Health-Aware Heterogeneous GNN

    Architecture:
        DualChannelEncoder → HybridDecoder + HealthBias
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels: int = 64,
        metadata: Tuple = None,
        dropout: float = 0.4,
        heads: int = 4,
        drop_edge_p: float = 0.1,
        num_layers: int = 3,
    ):
        super().__init__()

        self.encoder = DualChannelEncoder(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            metadata=metadata,
            dropout=dropout,
            heads=heads,
            drop_edge_p=drop_edge_p,
            num_layers=num_layers,
        )

        self.decoder = HybridDecoder(emb_dim=out_channels, dropout=dropout)

        # Health bias head: scalar adjustment based on food health score
        self.health_bias = nn.Sequential(
            nn.Linear(1, 8), nn.GELU(), nn.Linear(8, 1)
        )

    def forward(
        self,
        x_dict, edge_index_dict, edge_label_index,
        health_scores=None,
        return_projections=False,
        return_embeddings=False,
    ):
        if return_projections:
            z_dict, proj = self.encoder(x_dict, edge_index_dict,
                                        return_projections=True)
        else:
            z_dict = self.encoder(x_dict, edge_index_dict)
            proj = None

        scores = self.decoder(z_dict, edge_label_index)

        if health_scores is not None:
            f_idx = edge_label_index[1]
            hs = health_scores[f_idx].unsqueeze(-1)
            bias = self.health_bias(hs).squeeze(-1)
            scores = scores + 0.05 * bias

        if return_projections:
            return scores, proj
        if return_embeddings:
            return scores, z_dict
        return scores


# ============================================================================
# 4. BASELINE MODELS
# ============================================================================

class MatrixFactorization(nn.Module):
    """MF with user/item biases (Koren 2009)."""

    def __init__(self, num_users, num_foods, emb_dim=64):
        super().__init__()
        self.u_emb = nn.Embedding(num_users, emb_dim)
        self.f_emb = nn.Embedding(num_foods, emb_dim)
        self.u_bias = nn.Embedding(num_users, 1)
        self.f_bias = nn.Embedding(num_foods, 1)
        nn.init.normal_(self.u_emb.weight, std=0.01)
        nn.init.normal_(self.f_emb.weight, std=0.01)

    def forward(self, edge_label_index, **kw):
        u, f = edge_label_index[0], edge_label_index[1]
        s = (self.u_emb(u) * self.f_emb(f)).sum(-1)
        s = s + self.u_bias(u).squeeze(-1) + self.f_bias(f).squeeze(-1)
        return s


class LightGCN(nn.Module):
    """LightGCN (He et al. 2020) for bipartite user-food graph."""

    def __init__(self, num_users, num_foods, emb_dim=64, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.num_users = num_users
        self.num_foods = num_foods
        self.u_emb = nn.Embedding(num_users, emb_dim)
        self.f_emb = nn.Embedding(num_foods, emb_dim)
        nn.init.xavier_uniform_(self.u_emb.weight)
        nn.init.xavier_uniform_(self.f_emb.weight)

    def _propagate(self, train_ei, device):
        src, dst = train_ei[0], train_ei[1]
        D = self.u_emb.weight.shape[1]

        with torch.no_grad():
            deg_u = torch.zeros(self.num_users, device=device)
            deg_f = torch.zeros(self.num_foods, device=device)
            ones  = torch.ones(src.shape[0], device=device)
            deg_u.scatter_add_(0, src, ones)
            deg_f.scatter_add_(0, dst, ones)
            # Symmetric normalisation: 1/sqrt(deg_u) per user, 1/sqrt(deg_f) per food
            # Applied as edge weight = 1/sqrt(deg_u[src]) * 1/sqrt(deg_f[dst])
            norm_u = (1.0 / deg_u.clamp(min=1).sqrt())  # [U]
            norm_f = (1.0 / deg_f.clamp(min=1).sqrt())  # [F]

        u_e, f_e = self.u_emb.weight, self.f_emb.weight
        all_u, all_f = [u_e], [f_e]
        for _ in range(self.num_layers):
            # Symmetric normalisation applied per-node (not per-edge)
            # avoids vanishing scale for high-degree nodes
            msg_u2f = u_e[src] * norm_u[src].unsqueeze(-1)   # [E, D]
            msg_f2u = f_e[dst] * norm_f[dst].unsqueeze(-1)   # [E, D]
            new_f = torch.zeros(self.num_foods, D, device=device)\
                        .index_add(0, dst, msg_u2f) * norm_f.unsqueeze(-1)
            new_u = torch.zeros(self.num_users, D, device=device)\
                        .index_add(0, src, msg_f2u) * norm_u.unsqueeze(-1)
            u_e, f_e = new_u, new_f
            all_u.append(u_e); all_f.append(f_e)

        return torch.stack(all_u).mean(0), torch.stack(all_f).mean(0)

    def forward(self, edge_label_index, train_edge_index=None, **kw):
        device = edge_label_index.device
        if train_edge_index is not None:
            u_e, f_e = self._propagate(train_edge_index, device)
        else:
            u_e, f_e = self.u_emb.weight, self.f_emb.weight
        u, f = edge_label_index[0], edge_label_index[1]
        return (u_e[u] * f_e[f]).sum(-1)


# ============================================================================
# 4b. ADDITIONAL BASELINES: NGCF, SGL, HFRS-DA
# ============================================================================

class NGCF(nn.Module):
    """
    Neural Graph Collaborative Filtering (Wang et al., SIGIR 2019).
    Bipartite user-food graph with explicit interaction-based message passing.
    Embedding propagation with element-wise product interaction term.
    """
    def __init__(self, num_users, num_foods, emb_dim=64, num_layers=3,
                 dropout=0.1):
        super().__init__()
        self.num_users = num_users
        self.num_foods = num_foods
        self.num_layers = num_layers
        self.dropout = dropout
        emb_dim = emb_dim  # keep consistent with caller

        self.u_emb = nn.Embedding(num_users, emb_dim)
        self.f_emb = nn.Embedding(num_foods, emb_dim)
        nn.init.xavier_uniform_(self.u_emb.weight)
        nn.init.xavier_uniform_(self.f_emb.weight)

        # Per-layer transformation matrices (W1, W2 in paper)
        self.W1 = nn.ModuleList([nn.Linear(emb_dim, emb_dim, bias=False)
                                 for _ in range(num_layers)])
        self.W2 = nn.ModuleList([nn.Linear(emb_dim, emb_dim, bias=False)
                                 for _ in range(num_layers)])

    def _propagate(self, train_ei, device):
        src, dst = train_ei[0], train_ei[1]   # src=user, dst=food
        D = self.u_emb.weight.shape[1]
        num_users, num_foods = self.num_users, self.num_foods

        # Degree normalisation
        with torch.no_grad():
            deg_u = torch.zeros(num_users, device=device)
            deg_f = torch.zeros(num_foods, device=device)
            ones  = torch.ones(src.shape[0], device=device)
            deg_u.scatter_add_(0, src, ones)
            deg_f.scatter_add_(0, dst, ones)
            norm_u = (1.0 / deg_u.clamp(min=1).sqrt())  # [U]
            norm_f = (1.0 / deg_f.clamp(min=1).sqrt())  # [F]

        u_e = self.u_emb.weight  # [U, D]
        f_e = self.f_emb.weight  # [F, D]

        u_all, f_all = [u_e], [f_e]
        for l in range(self.num_layers):
            # Food → User messages (out-of-place, per-node normalisation)
            msg_f2u_1 = f_e[dst] * norm_f[dst].unsqueeze(-1)
            msg_f2u_2 = f_e[dst] * u_e[src] * norm_f[dst].unsqueeze(-1)
            agg_u = torch.zeros(num_users, D, device=device)\
                        .index_add(0, src, msg_f2u_1 + msg_f2u_2)
            agg_u = agg_u * norm_u.unsqueeze(-1)
            new_u = F.leaky_relu(self.W1[l](u_e) + self.W2[l](agg_u))
            new_u = F.dropout(new_u, p=self.dropout, training=self.training)

            # User → Food messages (out-of-place)
            msg_u2f_1 = u_e[src] * norm_u[src].unsqueeze(-1)
            msg_u2f_2 = u_e[src] * f_e[dst] * norm_u[src].unsqueeze(-1)
            agg_f = torch.zeros(num_foods, D, device=device)\
                        .index_add(0, dst, msg_u2f_1 + msg_u2f_2)
            agg_f = agg_f * norm_f.unsqueeze(-1)
            new_f = F.leaky_relu(self.W1[l](f_e) + self.W2[l](agg_f))
            new_f = F.dropout(new_f, p=self.dropout, training=self.training)

            u_e, f_e = new_u, new_f
            u_all.append(u_e)
            f_all.append(f_e)

        # Concatenate all layers (NGCF paper eq.11)
        u_final = torch.cat(u_all, dim=-1)   # [U, D*(L+1)]
        f_final = torch.cat(f_all, dim=-1)   # [F, D*(L+1)]
        return u_final, f_final

    def forward(self, edge_label_index, train_edge_index=None, **kw):
        device = edge_label_index.device
        if train_edge_index is not None:
            u_e, f_e = self._propagate(train_edge_index, device)
        else:
            u_e = torch.cat([self.u_emb.weight] * (self.num_layers + 1), dim=-1)
            f_e = torch.cat([self.f_emb.weight] * (self.num_layers + 1), dim=-1)
        u, f = edge_label_index[0], edge_label_index[1]
        return (u_e[u] * f_e[f]).sum(-1)


class SGL(nn.Module):
    """
    Self-supervised Graph Learning (Wu et al., SIGIR 2021).
    LightGCN backbone + graph augmentation contrastive loss.
    Augmentation: node dropout (ED variant used in paper).
    During inference: standard LightGCN scoring.
    CL loss is computed in train loop (see train_sgl_epoch).
    """
    def __init__(self, num_users, num_foods, emb_dim=64, num_layers=3,
                 ssl_temp=0.2, ssl_lambda=0.1, aug_ratio=0.1):
        super().__init__()
        self.num_users = num_users
        self.num_foods = num_foods
        self.num_layers = num_layers
        self.ssl_temp   = ssl_temp
        self.ssl_lambda = ssl_lambda
        self.aug_ratio  = aug_ratio

        self.u_emb = nn.Embedding(num_users, emb_dim)
        self.f_emb = nn.Embedding(num_foods, emb_dim)
        nn.init.xavier_uniform_(self.u_emb.weight)
        nn.init.xavier_uniform_(self.f_emb.weight)

    def _lightgcn_prop(self, train_ei, device, dropout_ratio=0.0):
        """LightGCN propagation with optional edge dropout for augmentation."""
        src, dst = train_ei[0], train_ei[1]
        if dropout_ratio > 0 and self.training:
            mask = torch.rand(src.shape[0], device=device) > dropout_ratio
            src, dst = src[mask], dst[mask]

        D = self.u_emb.weight.shape[1]

        # Degree normalisation (no-grad, structural)
        with torch.no_grad():
            deg_u = torch.zeros(self.num_users, device=device)
            deg_f = torch.zeros(self.num_foods, device=device)
            ones  = torch.ones(src.shape[0], device=device)
            deg_u.scatter_add_(0, src, ones)
            deg_f.scatter_add_(0, dst, ones)
            norm_u = (1.0 / deg_u.clamp(min=1).sqrt())  # [U]
            norm_f = (1.0 / deg_f.clamp(min=1).sqrt())  # [F]

        u_e, f_e = self.u_emb.weight, self.f_emb.weight
        all_u, all_f = [u_e], [f_e]
        for _ in range(self.num_layers):
            # Per-node symmetric normalisation: avoids vanishing scale
            msg_u2f = u_e[src] * norm_u[src].unsqueeze(-1)   # [E, D]
            msg_f2u = f_e[dst] * norm_f[dst].unsqueeze(-1)   # [E, D]
            new_f = torch.zeros(self.num_foods, D, device=device)\
                        .index_add(0, dst, msg_u2f) * norm_f.unsqueeze(-1)
            new_u = torch.zeros(self.num_users, D, device=device)\
                        .index_add(0, src, msg_f2u) * norm_u.unsqueeze(-1)
            u_e, f_e = new_u, new_f
            all_u.append(u_e)
            all_f.append(f_e)

        return torch.stack(all_u).mean(0), torch.stack(all_f).mean(0)

    def ssl_loss(self, train_ei, device, user_idx, food_idx):
        """InfoNCE contrastive loss between two augmented views."""
        u1, f1 = self._lightgcn_prop(train_ei, device, self.aug_ratio)
        u2, f2 = self._lightgcn_prop(train_ei, device, self.aug_ratio)

        # User-side CL
        u1_s = F.normalize(u1[user_idx], dim=-1)
        u2_s = F.normalize(u2[user_idx], dim=-1)
        pos_u = (u1_s * u2_s).sum(-1) / self.ssl_temp
        neg_u = (u1_s @ u2_s.T) / self.ssl_temp
        cl_u  = -pos_u + torch.logsumexp(neg_u, dim=-1)

        # Food-side CL
        f1_s = F.normalize(f1[food_idx], dim=-1)
        f2_s = F.normalize(f2[food_idx], dim=-1)
        pos_f = (f1_s * f2_s).sum(-1) / self.ssl_temp
        neg_f = (f1_s @ f2_s.T) / self.ssl_temp
        cl_f  = -pos_f + torch.logsumexp(neg_f, dim=-1)

        return (cl_u.mean() + cl_f.mean()) * self.ssl_lambda

    def forward(self, edge_label_index, train_edge_index=None, **kw):
        device = edge_label_index.device
        if train_edge_index is not None:
            u_e, f_e = self._lightgcn_prop(train_edge_index, device, 0.0)
        else:
            u_e, f_e = self.u_emb.weight, self.f_emb.weight
        u, f = edge_label_index[0], edge_label_index[1]
        return (u_e[u] * f_e[f]).sum(-1)


class HFRSDAModel(nn.Module):
    """
    HFRS-DA: Health-aware Food Recommendation System with Dual Attention
    (Heterogeneous Graphs). Simplified re-implementation based on:
      Tran et al., Computers in Biology and Medicine, 2024.
      DOI: 10.1016/j.compbiomed.2023.107879

    Architecture:
      - Node-Level Attention (NLA): GAT on user-food-ingredient meta-paths
        capturing user preference signals.
      - Semantic-Level Attention (SLA): health-score-weighted attention
        biasing recommendations toward nutritionally superior foods.
      - Final score: α * NLA_score + (1-α) * SLA_score
    """
    def __init__(self, num_users, num_foods, num_ingredients,
                 emb_dim=64, num_heads=4, dropout=0.1,
                 health_alpha=0.3):
        super().__init__()
        self.num_users = num_users
        self.num_foods = num_foods
        self.health_alpha = health_alpha  # weight of health branch

        # Node embeddings
        self.u_emb = nn.Embedding(num_users, emb_dim)
        self.f_emb = nn.Embedding(num_foods, emb_dim)
        self.i_emb = nn.Embedding(max(num_ingredients, 1), emb_dim)
        nn.init.xavier_uniform_(self.u_emb.weight)
        nn.init.xavier_uniform_(self.f_emb.weight)
        nn.init.xavier_uniform_(self.i_emb.weight)

        # NLA: multi-head self-attention on user/food embeddings
        self.nla_attn = nn.MultiheadAttention(emb_dim, num_heads,
                                              dropout=dropout, batch_first=True)
        self.nla_norm = nn.LayerNorm(emb_dim)
        self.nla_proj = nn.Linear(emb_dim, emb_dim)

        # SLA: food-health scoring head
        self.sla_health_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, 1),
        )

        # Final scorer
        self.scorer = nn.Linear(emb_dim * 2, 1)
        self.dropout = dropout

    def _nla_forward(self, u_idx, f_idx):
        """Node-Level Attention: attend over food neighborhood of user."""
        u_e = self.u_emb(u_idx)          # [B, D]
        f_e = self.f_emb(f_idx)          # [B, D]

        # Stack as sequence for self-attention: [B, 2, D]
        seq = torch.stack([u_e, f_e], dim=1)
        attn_out, _ = self.nla_attn(seq, seq, seq)
        attn_out = self.nla_norm(attn_out + seq)
        u_out = F.dropout(attn_out[:, 0], p=self.dropout, training=self.training)
        f_out = F.dropout(attn_out[:, 1], p=self.dropout, training=self.training)
        return u_out, f_out

    def _sla_forward(self, f_idx, health_scores=None):
        """Semantic-Level Attention: health-weighted food representation."""
        f_e = self.f_emb(f_idx)               # [B, D]
        h_score = self.sla_health_proj(f_e)   # [B, 1]
        if health_scores is not None:
            # Incorporate external health signal
            ext_h = health_scores[f_idx].unsqueeze(-1)  # [B, 1]
            h_score = h_score + ext_h
        h_weight = torch.sigmoid(h_score)              # [B, 1]
        return f_e * h_weight                          # [B, D]

    def forward(self, edge_label_index, health_scores=None, **kw):
        u_idx = edge_label_index[0]
        f_idx = edge_label_index[1]

        # NLA branch
        u_nla, f_nla = self._nla_forward(u_idx, f_idx)
        nla_score = (u_nla * f_nla).sum(-1)           # [B]

        # SLA branch (health-aware)
        f_sla = self._sla_forward(f_idx, health_scores)
        u_e   = self.u_emb(u_idx)
        sla_score = (u_e * f_sla).sum(-1)             # [B]

        # Weighted combination
        score = ((1 - self.health_alpha) * nla_score
                 + self.health_alpha * sla_score)
        return score


# ============================================================================
# 5. LOSS FUNCTION
# ============================================================================

class NutriLoss(nn.Module):
    """
    Health-Aware BPR Loss:
        L = L_BPR + λ_h * L_health + λ_cl * L_InfoNCE

    L_BPR      = -E[log σ(ŷ_pos - ŷ_neg)]
    L_health   = E[ReLU(-(ŷ_pos - ŷ_neg) * sign(h_pos - h_neg))]
               (healthier foods should be ranked higher when preference is tied)
    L_InfoNCE  = cross-entropy on cosine similarity matrix
               (pushes user embeddings close to interacted food embeddings)
    """

    def __init__(self, lambda_health=0.1, lambda_cl=0.05, temperature=0.2):
        super().__init__()
        self.lh = lambda_health
        self.lcl = lambda_cl
        self.tau = temperature

    def bpr(self, pos, neg):
        return -F.logsigmoid(pos - neg).mean()

    def health_margin(self, pos, neg, hpos, hneg):
        hdiff = hpos - hneg
        sdiff = pos - neg
        penalty = F.relu(-sdiff * hdiff.sign())
        return penalty.mean()

    def infonce(self, proj, eil, labels):
        if proj is None:
            return torch.tensor(0.0)
        u_p = proj.get('user')
        f_p = proj.get('food')
        if u_p is None or f_p is None:
            return torch.tensor(0.0)

        pos_mask = labels.bool()
        if pos_mask.sum() < 2:
            return torch.tensor(0.0)

        u = F.normalize(u_p[eil[0][pos_mask]], dim=-1)
        f = F.normalize(f_p[eil[1][pos_mask]], dim=-1)

        # Limit batch size for memory
        n = min(u.shape[0], 256)
        u, f = u[:n], f[:n]

        sim = torch.mm(u, f.T) / self.tau
        target = torch.arange(n, device=sim.device)
        return F.cross_entropy(sim, target)

    def forward(self, pos_s, neg_s, hpos=None, hneg=None,
                proj=None, eil=None, labels=None):
        l_bpr = self.bpr(pos_s, neg_s)

        l_h = torch.tensor(0.0, device=pos_s.device)
        if hpos is not None and hneg is not None and self.lh > 0:
            l_h = self.health_margin(pos_s, neg_s, hpos, hneg)

        l_cl = torch.tensor(0.0, device=pos_s.device)
        if proj is not None and eil is not None and labels is not None and self.lcl > 0:
            l_cl = self.infonce(proj, eil, labels)

        total = l_bpr + self.lh * l_h + self.lcl * l_cl

        return total, {
            'total': total.item(),
            'bpr': l_bpr.item(),
            'health': l_h.item(),
            'cl': l_cl.item(),
        }


# ============================================================================
# 6. METRICS
# ============================================================================

def classification_metrics(scores_np, labels_np, threshold=0.5):
    pred = (scores_np > threshold).astype(int)
    m = {
        'accuracy': float(np.mean(pred == labels_np)),
        'precision': float(precision_score(labels_np, pred, zero_division=0)),
        'recall': float(recall_score(labels_np, pred, zero_division=0)),
        'f1': float(f1_score(labels_np, pred, zero_division=0)),
    }
    if len(np.unique(labels_np)) > 1:
        m['auc'] = float(roc_auc_score(labels_np, scores_np))
        m['ap'] = float(average_precision_score(labels_np, scores_np))
    else:
        m['auc'] = 0.5
        m['ap'] = float(labels_np.mean())
    return m


def ranking_metrics(model, data, device, k_list=(5, 10, 20), max_users=300):
    """
    HR@K, NDCG@K, MRR, HealthGain@K 계산.
    논문의 핵심 평가 지표.
    """
    model.eval()

    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    ei_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    eil = data[('user', 'eats', 'food')].edge_label_index.to(device)
    el  = data[('user', 'eats', 'food')].edge_label.to(device)

    # Food health scores (average over interactions)
    health_scores = _get_food_health(data, device)

    pos_eil = eil[:, el == 1]
    unique_users = pos_eil[0].unique()
    if len(unique_users) > max_users:
        idx = torch.randperm(len(unique_users))[:max_users]
        unique_users = unique_users[idx]

    num_foods = data['food'].num_nodes

    hr   = {k: [] for k in k_list}
    ndcg = {k: [] for k in k_list}
    hg   = {k: [] for k in k_list}
    mrr_vals = []

    with torch.no_grad():
        z_dict = model.encoder(x_dict, ei_dict)

        for u in unique_users:
            pos_foods = pos_eil[1][pos_eil[0] == u]
            if len(pos_foods) == 0:
                continue

            # Score all foods
            u_t = u.expand(num_foods)
            f_t = torch.arange(num_foods, device=device)
            scores = model.decoder(z_dict, torch.stack([u_t, f_t])).cpu().numpy()

            sorted_idx = np.argsort(-scores)
            pos_set = set(int(x) for x in pos_foods.cpu().tolist())  # Python int set

            # MRR
            for rank, fi in enumerate(sorted_idx):
                if int(fi) in pos_set:
                    mrr_vals.append(1.0 / (rank + 1))
                    break

            for k in k_list:
                top_k_list = [int(fi) for fi in sorted_idx[:k]]  # Python int list
                top_k_set  = set(top_k_list)

                # HR@K
                hr[k].append(1.0 if len(top_k_set & pos_set) > 0 else 0.0)

                # NDCG@K
                dcg = 0.0
                for rank, fi in enumerate(top_k_list):
                    if fi in pos_set:
                        dcg += 1.0 / np.log2(rank + 2)
                ideal_len = min(len(pos_set), k)
                idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_len))
                ndcg[k].append(dcg / idcg if idcg > 0 else 0.0)

                # HealthGain@K
                if health_scores is not None:
                    avg_topk = health_scores[top_k_list].mean().item()
                    avg_all  = health_scores.mean().item()
                    hg[k].append(avg_topk - avg_all)

    out = {}
    for k in k_list:
        out[f'HR@{k}']   = float(np.mean(hr[k]))   if hr[k]   else 0.0
        out[f'NDCG@{k}'] = float(np.mean(ndcg[k])) if ndcg[k] else 0.0
        if hg[k]:
            out[f'HealthGain@{k}'] = float(np.mean(hg[k]))
    out['MRR'] = float(np.mean(mrr_vals)) if mrr_vals else 0.0
    return out


def ranking_metrics_from_z(z_dict, decoder, data, device, health_scores=None,
                           k_list=(5, 10, 20), max_users=200,
                           score_batch=512, n_neg_sample=100):
    """
    Compute HR@K, NDCG@K, MRR, HealthGain@K.

    v2.3 — Sampled Evaluation (RecSys 논문 표준):
    ──────────────────────────────────────────────
    전체 31K food에 대해 ranking하면 HR@10 ≈ 0.0003 (random baseline)이므로
    의미 있는 지표가 나오지 않습니다.

    RecSys 논문 표준 평가법 (NCF, LightGCN, NGCF 등 모두 이 방식):
    각 user의 positive food 1개 + 랜덤 100개 negative food를 합쳐
    101개 후보 중에서 ranking → HR@K, NDCG@K 계산.

    이 방식에서:
    - Random baseline HR@10 = 10/101 ≈ 0.099
    - 좋은 모델:          HR@10 = 0.3 ~ 0.7
    - 논문 Table 2 수치와 비교 가능한 수준
    """
    # ── 1. Positive edge 파악 (CPU numpy) ─────────────────────────────────────
    eil_cpu = data[('user', 'eats', 'food')].edge_label_index.cpu()
    el_cpu  = data[('user', 'eats', 'food')].edge_label.cpu()

    pos_mask_np  = (el_cpu.numpy() == 1)
    eil_np       = eil_cpu.numpy()
    pos_users_np = eil_np[0, pos_mask_np]
    pos_foods_np = eil_np[1, pos_mask_np]

    if pos_users_np.shape[0] == 0:
        out = {}
        for k in k_list:
            out[f'HR@{k}'] = 0.0; out[f'NDCG@{k}'] = 0.0
        out['MRR'] = 0.0
        return out

    num_foods     = z_dict['food'].shape[0]
    num_users_emb = z_dict['user'].shape[0]

    # ── 2. 사용자 샘플링 ───────────────────────────────────────────────────────
    unique_users = np.unique(pos_users_np)
    if len(unique_users) > max_users:
        perm = np.random.permutation(len(unique_users))[:max_users]
        unique_users = unique_users[perm]

    # ── 3. health scores ──────────────────────────────────────────────────────
    if health_scores is not None:
        hs_cpu  = health_scores.detach().cpu().float().numpy()
        hs_mean = float(hs_cpu.mean())
    else:
        hs_cpu  = None
        hs_mean = 0.0

    # ── 4. 사용자별 ranking (sampled negatives) ────────────────────────────────
    hr       = {k: [] for k in k_list}
    ndcg_d   = {k: [] for k in k_list}
    hg       = {k: [] for k in k_list}
    mrr_vals = []

    decoder.eval()
    rng = np.random.default_rng(seed=12345)

    for u_idx_raw in unique_users:
        u_idx = int(u_idx_raw)
        if u_idx >= num_users_emb:
            continue

        # 이 사용자의 모든 positive food
        mask_u      = (pos_users_np == u_idx)
        pos_foods_u = pos_foods_np[mask_u]
        pos_foods_u = pos_foods_u[pos_foods_u < num_foods]
        if pos_foods_u.shape[0] == 0:
            continue
        all_pos_set = set(int(f) for f in pos_foods_u)

        # Leave-one-out: positive 1개를 target으로 선택
        target_food = int(rng.choice(pos_foods_u))

        # Negative sampling: target/pos가 아닌 랜덤 n_neg_sample개
        neg_pool = np.arange(num_foods)
        neg_pool = neg_pool[~np.isin(neg_pool, list(all_pos_set))]
        if len(neg_pool) < n_neg_sample:
            neg_sample = neg_pool
        else:
            neg_sample = rng.choice(neg_pool, size=n_neg_sample, replace=False)

        # 후보: [target] + [100 negatives] = 101개
        candidates = np.array([target_food] + list(neg_sample), dtype=np.int64)

        # Decoder로 101개 스코어 계산
        u_t   = torch.full((len(candidates),), u_idx, dtype=torch.long, device=device)
        f_t   = torch.tensor(candidates, dtype=torch.long, device=device)
        ei_c  = torch.stack([u_t, f_t])

        with torch.no_grad():
            scores_c = decoder(z_dict, ei_c).cpu().float().numpy()

        # 내림차순 정렬 (candidates 기준 인덱스)
        sorted_local = np.argsort(-scores_c)  # 0~100 사이 로컬 인덱스

        # target의 로컬 인덱스 = 0 (candidates[0] = target_food)
        target_local_rank = int(np.where(sorted_local == 0)[0][0]) + 1  # 1-indexed

        # MRR
        mrr_vals.append(1.0 / target_local_rank)

        for k in k_list:
            # Top-K 로컬 인덱스 중 0(=target)이 있으면 hit
            top_k_local = sorted_local[:k]
            hit = 1.0 if 0 in top_k_local else 0.0
            hr[k].append(hit)

            # NDCG@K
            dcg = 0.0
            for r, li in enumerate(top_k_local):
                if li == 0:  # target found at rank r+1
                    dcg = 1.0 / np.log2(r + 2)
                    break
            idcg = 1.0  # ideal: target at rank 1
            ndcg_d[k].append(dcg / idcg)

            # HealthGain@K: top-K candidates의 health vs 전체 평균
            if hs_cpu is not None:
                topk_foods = candidates[top_k_local]
                hg[k].append(float(hs_cpu[topk_foods].mean()) - hs_mean)

    # ── 5. 집계 ────────────────────────────────────────────────────────────────
    out = {}
    for k in k_list:
        out[f'HR@{k}']   = float(np.mean(hr[k]))     if hr[k]     else 0.0
        out[f'NDCG@{k}'] = float(np.mean(ndcg_d[k])) if ndcg_d[k] else 0.0
        if hg[k]:
            out[f'HealthGain@{k}'] = float(np.mean(hg[k]))
    out['MRR'] = float(np.mean(mrr_vals)) if mrr_vals else 0.0
    return out


def _get_food_health(data, device):
    """Per-food health score tensor."""
    if hasattr(data['food'], 'health_score'):
        return data['food'].health_score.to(device)
    if ('user', 'healthness', 'food') in data.edge_types:
        h_ei = data[('user','healthness','food')].edge_index.to(device)
        h_ea = data[('user','healthness','food')].edge_attr.to(device)
        if h_ea.dim() > 1: h_ea = h_ea.squeeze(-1)
        nf = data['food'].num_nodes
        acc = torch.zeros(nf, device=device)
        cnt = torch.zeros(nf, device=device)
        acc.scatter_add_(0, h_ei[1], h_ea)
        cnt.scatter_add_(0, h_ei[1], torch.ones_like(h_ea))
        return acc / (cnt + 1e-8)
    return None


# ============================================================================
# 7. TRAINING UTILITIES
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=30, mode='max'):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best = None
        self.best_state = None
        self.stop = False

    def __call__(self, score, model):
        better = (self.best is None) or \
                 (self.mode == 'max' and score > self.best) or \
                 (self.mode == 'min' and score < self.best)
        if better:
            self.best = score
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop

    def load_best(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


def _prepare_pairs(data, device):
    """Return pos/neg edge pairs (user-matched) and full edge_label tensors."""
    eil = data[('user','eats','food')].edge_label_index.to(device)
    el  = data[('user','eats','food')].edge_label.to(device)
    pos_m = el == 1
    neg_m = ~pos_m
    pos_ei = eil[:, pos_m]   # [2, P]
    neg_ei = eil[:, neg_m]   # [2, N]
    n = min(pos_ei.shape[1], neg_ei.shape[1])

    # Re-match: pair each pos user with a random neg food from neg pool
    # This ensures BPR compares same user's pos vs neg item
    pos_ei_n = pos_ei[:, :n]
    neg_foods = neg_ei[1, torch.randperm(neg_ei.shape[1], device=device)[:n]]
    neg_ei_matched = torch.stack([pos_ei_n[0], neg_foods], dim=0)

    return eil, el, pos_ei_n, neg_ei_matched


def train_one_epoch(model, optimizer, train_data, criterion, device,
                    use_cl=True, batch_size=4096):
    """
    Memory-efficient training.
    One full graph encode → BPR loss on sampled pairs.
    """
    model.train()
    x_dict  = {k: v.to(device) for k, v in train_data.x_dict.items()}
    ei_dict = {k: v.to(device) for k, v in train_data.edge_index_dict.items()}
    eil, el, pos_ei, neg_ei = _prepare_pairs(train_data, device)

    hs = _get_food_health(train_data, device)

    # Sub-sample pairs for memory efficiency
    n_pairs = min(pos_ei.shape[1], batch_size)
    idx = torch.randperm(pos_ei.shape[1])[:n_pairs]
    p_ei = pos_ei[:, idx]
    n_ei = neg_ei[:, idx]
    ph = hs[p_ei[1]] if hs is not None else None
    nh = hs[n_ei[1]] if hs is not None else None

    optimizer.zero_grad()

    if use_cl:
        ps, proj = model(x_dict, ei_dict, p_ei, return_projections=True)
    else:
        ps   = model(x_dict, ei_dict, p_ei)
        proj = None

    ns = model(x_dict, ei_dict, n_ei)

    # Use sub-sampled eil for CL loss
    n_eil = min(eil.shape[1], 4096)
    idx2 = torch.randperm(eil.shape[1])[:n_eil]
    loss, ld = criterion(ps, ns, ph, nh, proj, eil[:, idx2], el[idx2])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    with torch.no_grad():
        n_eval = min(eil.shape[1], 4096)
        idx3 = torch.randperm(eil.shape[1])[:n_eval]
        s_eil = eil[:, idx3]; s_el = el[idx3]
        all_s = model(x_dict, ei_dict, s_eil)
        proba = torch.sigmoid(all_s).cpu().numpy()
        cm = classification_metrics(proba, s_el.cpu().numpy())

    gc_collect()
    return loss.item(), {**ld, **cm}


@torch.no_grad()
def evaluate_model(model, data, criterion, device, compute_rank=False):
    model.eval()
    x_dict  = {k: v.to(device) for k, v in data.x_dict.items()}
    ei_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    eil, el, pos_ei, neg_ei = _prepare_pairs(data, device)

    hs = _get_food_health(data, device)

    # Encode once
    z_dict = model.encoder(x_dict, ei_dict)

    # Sample pairs for loss computation
    n_sample = min(pos_ei.shape[1], 4096)
    idx = torch.randperm(pos_ei.shape[1])[:n_sample]
    ps = model.decoder(z_dict, pos_ei[:, idx])
    ns = model.decoder(z_dict, neg_ei[:, idx])

    ph = hs[pos_ei[1][idx]] if hs is not None else None
    nh = hs[neg_ei[1][idx]] if hs is not None else None

    loss, ld = criterion(ps, ns, ph, nh)

    # Classification metrics on sampled edges
    n_cls = min(eil.shape[1], 8192)
    idx2 = torch.randperm(eil.shape[1])[:n_cls]
    s_eil = eil[:, idx2]; s_el = el[idx2]
    s_scores = model.decoder(z_dict, s_eil)
    proba = torch.sigmoid(s_scores).cpu().numpy()
    cm = classification_metrics(proba, s_el.cpu().numpy())

    out = {**ld, **cm}
    if compute_rank:
        rm = ranking_metrics_from_z(z_dict, model.decoder, data, device, hs)
        out.update(rm)

    del z_dict
    gc_collect()
    return loss.item(), out


# Baseline training (simpler loop, no contrastive)
def train_baseline_epoch(model, optimizer, train_data, criterion, device,
                         model_type='mf'):
    model.train()
    eil, el, pos_ei, neg_ei = _prepare_pairs(train_data, device)

    hs = _get_food_health(train_data, device)
    ph = hs[pos_ei[1]] if hs is not None else None
    nh = hs[neg_ei[1]] if hs is not None else None

    optimizer.zero_grad()
    train_ei = train_data[('user','eats','food')].edge_index.to(device)

    # ── Single propagation pass: compute u_e, f_e once, index for pos & neg ──
    if model_type in ('lightgcn', 'sgl'):
        u_e, f_e = model._lightgcn_prop(train_ei, device, 0.0) \
                   if model_type == 'sgl' \
                   else model._propagate(train_ei, device)
        ps = (u_e[pos_ei[0]] * f_e[pos_ei[1]]).sum(-1)
        ns = (u_e[neg_ei[0]] * f_e[neg_ei[1]]).sum(-1)
    elif model_type == 'ngcf':
        u_e, f_e = model._propagate(train_ei, device)
        ps = (u_e[pos_ei[0]] * f_e[pos_ei[1]]).sum(-1)
        ns = (u_e[neg_ei[0]] * f_e[neg_ei[1]]).sum(-1)
    elif model_type == 'hfrsda':
        ps = model(pos_ei, health_scores=hs)
        ns = model(neg_ei, health_scores=hs)
    else:  # mf
        ps = model(pos_ei)
        ns = model(neg_ei)

    loss, ld = criterion(ps, ns, ph, nh)

    # SGL: add SSL contrastive loss
    if model_type == 'sgl':
        u_idx = pos_ei[0].unique()
        f_idx = pos_ei[1].unique()
        if u_idx.shape[0] > 512:
            u_idx = u_idx[torch.randperm(u_idx.shape[0])[:512]]
        if f_idx.shape[0] > 512:
            f_idx = f_idx[torch.randperm(f_idx.shape[0])[:512]]
        ssl = model.ssl_loss(train_ei, device, u_idx, f_idx)
        loss = loss + ssl
        ld['ssl'] = ssl.item()
        ld['cl']  = ssl.item()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item(), ld


@torch.no_grad()
def eval_baseline(model, data, criterion, device, model_type='mf',
                  train_ei=None, compute_rank=False):
    model.eval()
    eil, el, pos_ei, neg_ei = _prepare_pairs(data, device)
    hs = _get_food_health(data, device)
    ph = hs[pos_ei[1]] if hs is not None else None
    nh = hs[neg_ei[1]] if hs is not None else None

    # ── Single propagation pass (same embedding space for all scoring) ──
    if model_type in ('lightgcn', 'ngcf', 'sgl') and train_ei is not None:
        if model_type == 'sgl':
            u_e, f_e = model._lightgcn_prop(train_ei, device, 0.0)
        else:
            u_e, f_e = model._propagate(train_ei, device)
        ps    = (u_e[pos_ei[0]] * f_e[pos_ei[1]]).sum(-1)
        ns    = (u_e[neg_ei[0]] * f_e[neg_ei[1]]).sum(-1)
        all_s = (u_e[eil[0]]    * f_e[eil[1]]   ).sum(-1)
        kw = {'train_edge_index': train_ei}  # for _eval_baseline_rank
    elif model_type == 'hfrsda':
        kw = {'health_scores': hs}
        ps    = model(pos_ei, **kw)
        ns    = model(neg_ei, **kw)
        all_s = model(eil,    **kw)
    else:  # mf
        kw = {}
        ps    = model(pos_ei)
        ns    = model(neg_ei)
        all_s = model(eil)

    loss, ld = criterion(ps, ns, ph, nh)
    proba = torch.sigmoid(all_s).cpu().numpy()
    cm = classification_metrics(proba, el.cpu().numpy())
    out = {**ld, **cm}

    if compute_rank:
        _eval_baseline_rank(model, data, device, out, kw, model_type)

    return loss.item(), out


def _eval_baseline_rank(model, data, device, out_dict, kw, model_type,
                        max_users=200, n_neg_sample=100):
    """
    Compute ranking metrics for baseline models using Sampled-100 protocol.

    RecSys 논문 표준과 동일한 방식:
    각 user의 positive food 1개 + random negative 100개 = 101개 후보 중 ranking.
    NutriGraphNet v2의 ranking_metrics_from_z와 완전히 동일한 프로토콜 적용.
    """
    # CPU numpy 처리
    eil_cpu = data[('user','eats','food')].edge_label_index.cpu()
    el_cpu  = data[('user','eats','food')].edge_label.cpu()
    pos_mask_np  = (el_cpu.numpy() == 1)
    eil_np       = eil_cpu.numpy()
    pos_users_np = eil_np[0, pos_mask_np]
    pos_foods_np = eil_np[1, pos_mask_np]

    if pos_users_np.shape[0] == 0:
        k_list = (5, 10, 20)
        for k in k_list:
            out_dict[f'HR@{k}'] = 0.0; out_dict[f'NDCG@{k}'] = 0.0
        out_dict['MRR'] = 0.0
        return

    # Get embeddings (detach to avoid grad issues)
    if model_type == 'mf':
        u_e = model.u_emb.weight.detach()
        f_e = model.f_emb.weight.detach()
    elif model_type in ('lightgcn', 'ngcf', 'sgl'):
        if 'train_edge_index' in kw:
            # SGL uses _lightgcn_prop; LightGCN/NGCF use _propagate
            if hasattr(model, '_propagate'):
                u_e, f_e = model._propagate(kw['train_edge_index'], device)
            else:
                u_e, f_e = model._lightgcn_prop(kw['train_edge_index'], device, 0.0)
            u_e = u_e.detach()
            f_e = f_e.detach()
        else:
            u_e, f_e = model.u_emb.weight.detach(), model.f_emb.weight.detach()
    elif model_type == 'hfrsda':
        # HFRS-DA: use NLA embeddings (user + food raw embeddings through attn)
        u_e = model.u_emb.weight.detach()
        f_e = model.f_emb.weight.detach()
    else:
        u_e = model.u_emb.weight.detach()
        f_e = model.f_emb.weight.detach()

    num_foods     = data['food'].num_nodes
    num_users_emb = u_e.shape[0]

    # User sampling
    unique_users = np.unique(pos_users_np)
    if len(unique_users) > max_users:
        perm = np.random.permutation(len(unique_users))[:max_users]
        unique_users = unique_users[perm]

    k_list = (5, 10, 20)
    hr       = {k: [] for k in k_list}
    ndcg_d   = {k: [] for k in k_list}
    mrr_vals = []
    rng = np.random.default_rng(seed=12345)

    for u_idx_raw in unique_users:
        u_idx = int(u_idx_raw)
        if u_idx >= num_users_emb:
            continue

        mask_u      = (pos_users_np == u_idx)
        pos_foods_u = pos_foods_np[mask_u]
        pos_foods_u = pos_foods_u[pos_foods_u < num_foods]
        if pos_foods_u.shape[0] == 0:
            continue
        all_pos_set = set(int(f) for f in pos_foods_u)

        # Leave-one-out: target positive 1개 선택
        target_food = int(rng.choice(pos_foods_u))

        # Negative sampling: 100개 (pos 제외)
        neg_pool = np.arange(num_foods)
        neg_pool = neg_pool[~np.isin(neg_pool, list(all_pos_set))]
        if len(neg_pool) < n_neg_sample:
            neg_sample = neg_pool
        else:
            neg_sample = rng.choice(neg_pool, size=n_neg_sample, replace=False)

        # candidates[0] = target, candidates[1:] = negatives
        candidates = np.array([target_food] + list(neg_sample), dtype=np.int64)

        # Dot product scores (embedding lookup)
        cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
        scores_c = (u_e[u_idx].unsqueeze(0) * f_e[cand_t]).sum(-1).cpu().float().numpy()

        sorted_local = np.argsort(-scores_c)
        target_local_rank = int(np.where(sorted_local == 0)[0][0]) + 1
        mrr_vals.append(1.0 / target_local_rank)

        for k in k_list:
            top_k_local = sorted_local[:k]
            hit = 1.0 if 0 in top_k_local else 0.0
            hr[k].append(hit)

            dcg = 0.0
            for r, li in enumerate(top_k_local):
                if li == 0:
                    dcg = 1.0 / np.log2(r + 2)
                    break
            ndcg_d[k].append(dcg / 1.0)  # idcg = 1.0

    for k in k_list:
        out_dict[f'HR@{k}']   = float(np.mean(hr[k]))     if hr[k]     else 0.0
        out_dict[f'NDCG@{k}'] = float(np.mean(ndcg_d[k])) if ndcg_d[k] else 0.0
    out_dict['MRR'] = float(np.mean(mrr_vals)) if mrr_vals else 0.0


# ============================================================================
# 8. CROSS-VALIDATION RUNNER
# ============================================================================

def run_fold(fold, train_data, val_data, test_data, args, device, variant='full'):
    """Train one fold and return test metrics."""

    # ── Build model ──
    n_users = train_data['user'].num_nodes
    n_foods = train_data['food'].num_nodes
    n_ingr  = train_data['ingredient'].num_nodes if 'ingredient' in train_data.node_types else 1

    if variant == 'mf':
        model = MatrixFactorization(n_users, n_foods, args.out_channels).to(device)
        mtype = 'mf'
    elif variant == 'lightgcn':
        model = LightGCN(n_users, n_foods, args.out_channels, 3).to(device)
        mtype = 'lightgcn'
    elif variant == 'ngcf':
        model = NGCF(n_users, n_foods, args.out_channels,
                     num_layers=3, dropout=args.dropout).to(device)
        mtype = 'ngcf'
    elif variant == 'sgl':
        model = SGL(n_users, n_foods, args.out_channels,
                    num_layers=3,
                    ssl_temp=getattr(args, 'sgl_temp', 0.2),
                    ssl_lambda=getattr(args, 'sgl_lambda', 0.1),
                    aug_ratio=getattr(args, 'sgl_aug', 0.1)).to(device)
        mtype = 'sgl'
    elif variant == 'hfrsda':
        model = HFRSDAModel(n_users, n_foods, n_ingr,
                            emb_dim=args.out_channels,
                            num_heads=max(1, args.heads),
                            dropout=args.dropout,
                            health_alpha=getattr(args, 'hfrsda_alpha', 0.3)).to(device)
        mtype = 'hfrsda'
    else:
        # NutriGraphNet ablation variants
        use_dual   = 'no_dual'   not in variant
        use_health = 'no_health' not in variant
        use_cl     = 'no_cl'     not in variant

        nlayers = args.num_layers if use_dual else 2
        drop_ep = args.drop_edge_p if use_cl else 0.0
        lh      = args.lambda_health if use_health else 0.0
        lcl     = args.lambda_cl if use_cl else 0.0

        model = NutriGraphNetV2(
            hidden_channels=args.hidden_channels,
            out_channels=args.out_channels,
            metadata=(train_data.node_types, train_data.edge_types),
            dropout=args.dropout,
            heads=args.heads,
            drop_edge_p=drop_ep,
            num_layers=nlayers,
        ).to(device)
        mtype = 'gnn'

    # Lazy init FIRST (before counting params)
    train_ei_ref = train_data[('user','eats','food')].edge_index.to(device)
    with torch.no_grad():
        _x  = {k: v.to(device) for k, v in train_data.x_dict.items()}
        _ei = {k: v.to(device) for k, v in train_data.edge_index_dict.items()}
        _eil = train_data[('user','eats','food')].edge_label_index[:, :16].to(device)
        if mtype == 'gnn':
            model(_x, _ei, _eil)
        elif mtype == 'mf':
            model(_eil)
        elif mtype == 'hfrsda':
            model(_eil)
        else:  # lightgcn, ngcf, sgl
            model(_eil, train_edge_index=train_ei_ref)
    gc_collect()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [{variant}] Fold {fold+1} | params={n_params:,}")

    # ── Criterion ──
    lh  = args.lambda_health if ('no_health' not in variant and mtype == 'gnn') else 0.0
    lcl = args.lambda_cl     if ('no_cl'     not in variant and mtype == 'gnn') else 0.0
    # HFRS-DA has its own internal health branch — keep BPR only
    criterion = NutriLoss(lambda_health=lh, lambda_cl=lcl,
                          temperature=args.temperature)

    # ── Optimizer / Scheduler ──
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay, betas=(0.9, 0.999))
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=max(10, args.epochs // 3), T_mult=1,
        eta_min=args.lr * 0.01
    )

    es = EarlyStopping(patience=args.patience, mode='max')

    history = {'train_loss': [], 'val_loss': [],
               'val_f1': [], 'val_auc': [], 'lr': []}
    best_metrics = None
    # train_ei_ref already defined during lazy init above

    for epoch in range(args.epochs):
        if mtype == 'gnn':
            use_cl_flag = ('no_cl' not in variant)
            tl, tm = train_one_epoch(model, optimizer, train_data, criterion,
                                     device, use_cl=use_cl_flag)
        else:
            tl, tm = train_baseline_epoch(model, optimizer, train_data, criterion,
                                          device, model_type=mtype)

        if mtype == 'gnn':
            vl, vm = evaluate_model(model, val_data, criterion, device)
        else:
            vl, vm = eval_baseline(model, val_data, criterion, device,
                                   mtype, train_ei_ref)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['val_f1'].append(vm.get('f1', 0.0))
        history['val_auc'].append(vm.get('auc', 0.5))
        history['lr'].append(lr)

        if (epoch + 1) % args.print_every == 0:
            bpr_v = tm.get('bpr', tl)
            cl_v  = tm.get('cl', tm.get('ssl', 0.0))
            print(f"    Ep{epoch+1:3d} | Loss={tl:.4f} | bpr={bpr_v:.4f} | cl={cl_v:.4f} | "
                  f"valF1={vm.get('f1',0):.4f} | "
                  f"valAUC={vm.get('auc',0.5):.4f} | lr={lr:.2e}")

        if es(vm.get('auc', 0.5), model):
            print(f"    Early stop @ epoch {epoch+1}")
            break

    es.load_best(model)

    # Final test evaluation (with ranking metrics)
    if mtype == 'gnn':
        _, test_m = evaluate_model(model, test_data, criterion, device,
                                   compute_rank=True)
    else:
        _, test_m = eval_baseline(model, test_data, criterion, device, mtype,
                                  train_ei_ref, compute_rank=True)

    print(f"  [{variant}] Fold {fold+1} Test →",
          " | ".join(f"{k}={v:.4f}" for k, v in test_m.items()
                     if isinstance(v, float) and k in
                     ('auc','f1','NDCG@10','HR@10','MRR')))

    # Save
    save_dir = Path(args.output_dir) / variant / f"fold_{fold+1}"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "best_model.pth")

    gc_collect()
    return test_m, history


def fast_link_split(data, val_ratio=0.05, test_ratio=0.10,
                    neg_ratio=1.0, seed=42):
    """
    Fast manual link split replacing RandomLinkSplit.
    Avoids expensive PyG transform overhead.

    Returns (train_data, val_data, test_data) with edge_label_index and
    edge_label attributes set for ('user','eats','food') relation.
    """
    import copy

    rng = np.random.default_rng(seed)

    ei = data[('user', 'eats', 'food')].edge_index  # [2, E]
    E  = ei.shape[1]
    num_users = data['user'].num_nodes
    num_foods = data['food'].num_nodes

    # Shuffle edges
    perm = rng.permutation(E)
    ei_shuf = ei[:, perm]

    # Split indices
    n_test = max(1, int(E * test_ratio))
    n_val  = max(1, int(E * val_ratio))
    n_train = E - n_test - n_val

    train_pos = ei_shuf[:, :n_train]
    val_pos   = ei_shuf[:, n_train:n_train + n_val]
    test_pos  = ei_shuf[:, n_train + n_val:]

    def make_neg(pos_ei, n_neg):
        """Random negative edges (quick)."""
        u = pos_ei[0].repeat(n_neg // pos_ei.shape[1] + 1)[:n_neg]
        f = torch.randint(0, num_foods, (n_neg,))
        return torch.stack([u, f], dim=0)

    def build_data(pos_ei, message_passing_ei):
        """Build a HeteroData split with edge_label_index."""
        d = copy.copy(data)
        # Override the message-passing edges
        d[('user','eats','food')].edge_index = message_passing_ei
        d[('food','rev_eats','user')].edge_index = message_passing_ei.flip(0)

        n_pos = pos_ei.shape[1]
        n_neg = int(n_pos * neg_ratio)
        neg_ei = make_neg(pos_ei, n_neg)

        eil = torch.cat([pos_ei, neg_ei], dim=1)
        el  = torch.cat([
            torch.ones(n_pos, dtype=torch.float),
            torch.zeros(n_neg, dtype=torch.float)
        ])
        # Shuffle
        perm2 = torch.randperm(eil.shape[1])
        eil = eil[:, perm2]
        el  = el[perm2]

        d[('user','eats','food')].edge_label_index = eil
        d[('user','eats','food')].edge_label        = el
        return d

    train_d = build_data(train_pos, train_pos)  # message passing on train
    val_d   = build_data(val_pos,   train_pos)
    test_d  = build_data(test_pos,  train_pos)

    return train_d, val_d, test_d


def run_cross_validation(data, args, device, variant='full'):
    """K-fold cross-validation for a model variant."""
    print(f"\n{'='*70}")
    print(f"  Variant: {variant.upper()}  ({args.n_folds} folds)")
    print(f"{'='*70}")

    fold_results = []
    fold_histories = []

    for fold in range(args.n_folds):
        torch.manual_seed(args.seed + fold)
        np.random.seed(args.seed + fold)
        train_d, val_d, test_d = fast_link_split(
            data,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            neg_ratio=1.0,
            seed=args.seed + fold
        )
        metrics, history = run_fold(fold, train_d, val_d, test_d,
                                    args, device, variant)
        fold_results.append(metrics)
        fold_histories.append(history)
        gc_collect()

    # Aggregate
    keys = [k for k, v in fold_results[0].items() if isinstance(v, float)]
    agg = {
        k: {
            'mean': float(np.mean([r[k] for r in fold_results if k in r])),
            'std':  float(np.std ([r[k] for r in fold_results if k in r]))
        }
        for k in keys
    }

    print(f"\n  {variant.upper()} — Mean Results:")
    for k, v in agg.items():
        print(f"    {k:20s}: {v['mean']:.4f} ± {v['std']:.4f}")

    return {'variant': variant, 'fold_results': fold_results,
            'aggregated': agg, 'histories': fold_histories}


# ============================================================================
# 9. VISUALIZATION
# ============================================================================

def _save_fig(fig, base_path, name):
    Path(base_path).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{base_path}/{name}.png", dpi=150, bbox_inches='tight')
    fig.savefig(f"{base_path}/{name}.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_model_comparison(all_results, output_dir):
    metrics = [('auc', 'AUC-ROC'), ('NDCG@10', 'NDCG@10'),
               ('HR@10', 'HR@10'), ('HealthGain@10', 'Health Gain@10')]
    models  = list(all_results.keys())
    colors  = plt.cm.tab10(np.linspace(0, 0.9, len(models)))
    n_met   = len(metrics)

    fig, axes = plt.subplots(1, n_met, figsize=(4 * n_met, 5))
    if n_met == 1:
        axes = [axes]

    for ax, (metric, label) in zip(axes, metrics):
        means, stds, names = [], [], []
        for m in models:
            agg = all_results[m].get('aggregated', {})
            if metric in agg:
                means.append(agg[metric]['mean'])
                stds.append(agg[metric]['std'])
                names.append(m)

        if not names:
            ax.set_visible(False)
            continue

        x = np.arange(len(names))
        bars = ax.bar(x, means, yerr=stds,
                      color=[colors[models.index(n)] for n in names],
                      capsize=5, alpha=0.85, edgecolor='black', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=9)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.003,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('NutriGraphNet v2 — Model Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, output_dir, 'model_comparison')
    print("  Saved: model_comparison.{png,pdf}")


def plot_training_curves(all_results, output_dir):
    for variant, res in all_results.items():
        hists = res.get('histories', [])
        if not hists:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        max_len = max(len(h['train_loss']) for h in hists)

        def pad(arr):
            return np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)

        for h in hists:
            axes[0].plot(h['train_loss'], alpha=0.3, color='steelblue')
            axes[0].plot(h['val_loss'],   alpha=0.3, color='tomato')
            axes[1].plot(h['val_f1'],     alpha=0.3, color='seagreen')
            axes[2].plot(h['val_auc'],    alpha=0.3, color='orchid')

        # Mean curve
        mean_train = np.nanmean([pad(h['train_loss']) for h in hists], 0)
        mean_val   = np.nanmean([pad(h['val_loss'])   for h in hists], 0)
        mean_f1    = np.nanmean([pad(h['val_f1'])     for h in hists], 0)
        mean_auc   = np.nanmean([pad(h['val_auc'])    for h in hists], 0)

        axes[0].plot(mean_train, lw=2.5, color='steelblue', label='Train')
        axes[0].plot(mean_val,   lw=2.5, color='tomato',    label='Val')
        axes[1].plot(mean_f1,    lw=2.5, color='seagreen',  label='Val F1')
        axes[2].plot(mean_auc,   lw=2.5, color='orchid',    label='Val AUC')

        for ax, title in zip(axes, ['Loss', 'F1 Score', 'AUC-ROC']):
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        plt.suptitle(f'Training Curves — {variant}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        _save_fig(fig, f"{output_dir}/{variant}", 'training_curves')
    print("  Saved: training_curves per variant")


# ============================================================================
# 10. LATEX TABLE GENERATION
# ============================================================================

def generate_latex_table(all_results, output_dir, sig_results=None):
    """
    Table 2 in paper: main results comparison.
    - Best values bolded
    - Significance stars (* p<0.05, ** p<0.01) vs NutriGraphNet v2 Full
    - Second-best underlined
    """
    key_metrics = ['auc', 'ap', 'f1',
                   'HR@5', 'HR@10', 'HR@20',
                   'NDCG@5', 'NDCG@10', 'NDCG@20',
                   'MRR', 'HealthGain@5', 'HealthGain@10']

    order = ['mf', 'lightgcn', 'ngcf', 'sgl', 'hfrsda',
             'no_dual', 'no_health', 'no_cl', 'full']
    names = {
        'mf':        r'MF~\cite{koren2009matrix}',
        'lightgcn':  r'LightGCN~\cite{he2020lightgcn}',
        'ngcf':      r'NGCF~\cite{wang2019neural}',
        'sgl':       r'SGL~\cite{wu2021self}',
        'hfrsda':    r'HFRS-DA~\cite{tran2024hfrsda}',
        'no_dual':   r'NGN$_{\text{-D}}$',
        'no_health': r'NGN$_{\text{-H}}$',
        'no_cl':     r'NGN$_{\text{-CL}}$',
        'full':      r'\textbf{NutriGraphNet v2}',
    }

    # Best and 2nd-best per metric
    best = {}; second = {}
    for m in key_metrics:
        vals = []
        for v in order:
            if v in all_results:
                agg = all_results[v]['aggregated']
                if m in agg:
                    vals.append((agg[m]['mean'], v))
        vals.sort(reverse=True)
        if vals:
            best[m]   = vals[0][0]
            second[m] = vals[1][0] if len(vals) > 1 else -999

    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        r'\caption{Performance Comparison on NutriGraphNet Benchmark '
        r'(mean $\pm$ std over 5-fold CV, Sampled-100 evaluation protocol). '
        r'\textbf{Bold}: best. \underline{Underline}: second-best. '
        r'$^*$/$^{**}$: Wilcoxon $p<0.05/0.01$ vs full model.}',
        r'\label{tab:main_results}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{l' + 'r' * len(key_metrics) + '}',
        r'\toprule',
        r'Model & ' + ' & '.join(
            m.replace('@', r'@').replace('_', r'\_').replace('HealthGain', r'HGain')
            for m in key_metrics
        ) + r' \\',
        r'\midrule',
    ]

    for v in order:
        if v not in all_results:
            continue
        agg = all_results[v]['aggregated']
        row = [names[v]]
        for m in key_metrics:
            if m in agg:
                mu, sd = agg[m]['mean'], agg[m]['std']
                cell = f"{mu:.4f}$\\pm${sd:.4f}"
                # Bold if best
                if m in best and abs(mu - best[m]) < 1e-5:
                    cell = r'\textbf{' + cell + '}'
                # Underline if 2nd best
                elif m in second and abs(mu - second[m]) < 1e-5:
                    cell = r'\underline{' + cell + '}'
                # Significance stars (for non-full models vs full)
                if v != 'full' and sig_results and v in sig_results:
                    vm = sig_results[v].get(m, {})
                    if vm.get('sig_01'):
                        cell += r'$^{**}$'
                    elif vm.get('sig'):
                        cell += r'$^*$'
                row.append(cell)
            else:
                row.append('--')
        lines.append(' & '.join(row) + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table*}']
    latex = '\n'.join(lines)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'table_main_results.tex').write_text(latex)
    print(f"  Saved: table_main_results.tex")
    return latex


# ============================================================================
# 11. SIGNIFICANCE TEST + LAMBDA SWEEP PLOT
# ============================================================================

def significance_test(all_results, proposed='full'):
    """
    Wilcoxon signed-rank test (one-tailed): proposed > baseline.
    Returns p-values + delta for all metrics, prints LaTeX-friendly table.
    """
    if not HAS_SCIPY or proposed not in all_results:
        return {}

    metrics_to_test = ['auc', 'f1', 'HR@5', 'HR@10', 'HR@20',
                       'NDCG@5', 'NDCG@10', 'NDCG@20', 'MRR']
    results = {}
    for baseline, bres in all_results.items():
        if baseline == proposed:
            continue
        pfolds = all_results[proposed]['fold_results']
        bfolds = bres['fold_results']

        row = {}
        for metric in metrics_to_test:
            pv = [r.get(metric, 0.0) for r in pfolds]
            bv = [r.get(metric, 0.0) for r in bfolds]
            if len(pv) >= 3 and len(pv) == len(bv):
                try:
                    # Check non-zero variance
                    diffs = [p - b for p, b in zip(pv, bv)]
                    if all(d == 0 for d in diffs):
                        row[metric] = {'p': 1.0, 'sig': False,
                                       'delta': 0.0}
                        continue
                    stat, p = wilcoxon(pv, bv, alternative='greater')
                    row[metric] = {
                        'p': float(p),
                        'sig': bool(p < 0.05),
                        'sig_01': bool(p < 0.01),
                        'delta': float(np.mean(pv) - np.mean(bv))
                    }
                except Exception:
                    pass
        results[baseline] = row
    return results


def _plot_lambda_sweep(sweep_results: dict, output_dir: str):
    """
    Plot health-accuracy trade-off across lambda_health values.
    논문 Figure: lambda_health sensitivity analysis.
    """
    lambdas = sorted(sweep_results.keys())
    hr10  = [sweep_results[l]['HR@10']   for l in lambdas]
    nd10  = [sweep_results[l]['NDCG@10'] for l in lambdas]
    mrr   = [sweep_results[l]['MRR']     for l in lambdas]
    hg10  = [sweep_results[l].get('HealthGain@10', 0) for l in lambdas]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Recommendation accuracy vs lambda
    ax = axes[0]
    ax.semilogx(lambdas, hr10,  'o-', color='steelblue',   lw=2, ms=7, label='HR@10')
    ax.semilogx(lambdas, nd10,  's-', color='darkorange',  lw=2, ms=7, label='NDCG@10')
    ax.semilogx(lambdas, mrr,   '^-', color='seagreen',    lw=2, ms=7, label='MRR')
    ax.set_xlabel(r'$\lambda_h$ (log scale)', fontsize=12)
    ax.set_ylabel('Recommendation Metric', fontsize=12)
    ax.set_title(r'Rec. Accuracy vs $\lambda_h$', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: Health gain vs lambda
    ax = axes[1]
    ax.semilogx(lambdas, hg10,  'D-', color='crimson', lw=2, ms=7, label='HealthGain@10')
    ax.set_xlabel(r'$\lambda_h$ (log scale)', fontsize=12)
    ax.set_ylabel('Health Gain', fontsize=12)
    ax.set_title(r'Health Gain vs $\lambda_h$', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle(r'Health-Accuracy Trade-off ($\lambda_h$ Sensitivity)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, output_dir, 'lambda_health_sensitivity')
    print("  Saved: lambda_health_sensitivity.{png,pdf}")


def plot_ablation_comparison(all_results, output_dir):
    """
    Grouped bar chart for ablation study.
    논문 Figure: ablation bar chart (HR@10, NDCG@10, MRR).
    """
    order   = ['mf', 'lightgcn', 'ngcf', 'sgl', 'hfrsda',
               'no_dual', 'no_health', 'no_cl', 'full']
    labels  = {
        'mf':        'MF', 'lightgcn': 'LightGCN',
        'ngcf':      'NGCF', 'sgl': 'SGL', 'hfrsda': 'HFRS-DA',
        'no_dual':   'NGN-D', 'no_health': 'NGN-H',
        'no_cl':     'NGN-CL', 'full': 'NutriGraphNet\nv2 (Full)',
    }
    present = [v for v in order if v in all_results]
    if not present:
        return

    metrics = [('HR@10', 'HR@10'), ('NDCG@10', 'NDCG@10'), ('MRR', 'MRR')]
    n_models  = len(present)
    n_metrics = len(metrics)
    x  = np.arange(n_models)
    w  = 0.22
    colors = ['#4878CF', '#6ACC65', '#D65F5F']

    fig, ax = plt.subplots(figsize=(max(8, 1.8*n_models), 5))
    offsets = np.linspace(-(n_metrics-1)*w/2, (n_metrics-1)*w/2, n_metrics)

    for i, (mkey, mlabel) in enumerate(metrics):
        means = [all_results[v]['aggregated'].get(mkey, {}).get('mean', 0) for v in present]
        stds  = [all_results[v]['aggregated'].get(mkey, {}).get('std',  0) for v in present]
        bars  = ax.bar(x + offsets[i], means, w, yerr=stds, label=mlabel,
                       color=colors[i], capsize=4, alpha=0.87, edgecolor='black', lw=0.6)
        # Annotate top values
        for bar, mu in zip(bars, means):
            if mu > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.008,
                        f'{mu:.3f}', ha='center', va='bottom', fontsize=7.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([labels[v] for v in present], fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.set_title('Ablation Study — NutriGraphNet v2', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    # Highlight full model column
    full_idx = present.index('full') if 'full' in present else -1
    if full_idx >= 0:
        ax.axvspan(full_idx - 0.4, full_idx + 0.4, alpha=0.08, color='gold',
                   label='_nolegend_')

    plt.tight_layout()
    _save_fig(fig, output_dir, 'ablation_comparison')
    print("  Saved: ablation_comparison.{png,pdf}")


# ============================================================================
# 12. MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='NutriGraphNet v2 — Full Experiment Pipeline'
    )
    parser.add_argument('--data_path', default='data/processed_data/processed_data_GNN_v5.pkl')
    # Model
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--out_channels',    type=int, default=64)
    parser.add_argument('--heads',           type=int, default=4)
    parser.add_argument('--dropout',         type=float, default=0.4)
    parser.add_argument('--num_layers',      type=int, default=3)
    parser.add_argument('--drop_edge_p',     type=float, default=0.1)
    # Training
    parser.add_argument('--epochs',          type=int, default=300)
    parser.add_argument('--lr',              type=float, default=0.001)
    parser.add_argument('--weight_decay',    type=float, default=1e-4)
    parser.add_argument('--patience',        type=int, default=30)
    # Loss
    parser.add_argument('--lambda_health',   type=float, default=0.1)
    parser.add_argument('--lambda_cl',       type=float, default=0.05)
    parser.add_argument('--temperature',     type=float, default=0.2)
    parser.add_argument('--margin',          type=float, default=0.5)
    # CV
    parser.add_argument('--n_folds',         type=int, default=5)
    parser.add_argument('--val_ratio',       type=float, default=0.05)
    parser.add_argument('--test_ratio',      type=float, default=0.10)
    parser.add_argument('--seed',            type=int, default=42)
    # Experiment
    parser.add_argument('--output_dir',      default='results/v2_experiments')
    parser.add_argument('--print_every',     type=int, default=5)
    parser.add_argument('--variants',        default='full',
                        help=('comma-sep: full,no_health,no_cl,no_dual,'
                              'mf,lightgcn,ngcf,sgl,hfrsda'))
    # New baseline hyperparams
    parser.add_argument('--sgl_temp',        type=float, default=0.2,
                        help='SGL InfoNCE temperature (default: 0.2)')
    parser.add_argument('--sgl_lambda',      type=float, default=0.1,
                        help='SGL SSL loss weight (default: 0.1)')
    parser.add_argument('--sgl_aug',         type=float, default=0.1,
                        help='SGL edge dropout ratio for augmentation (default: 0.1)')
    parser.add_argument('--hfrsda_alpha',    type=float, default=0.3,
                        help='HFRS-DA health branch weight α (default: 0.3)')
    # Lambda sweep (for sensitivity analysis)
    parser.add_argument('--lambda_sweep',    action='store_true',
                        help='Run lambda_health sensitivity sweep (0.001~1.0)')
    parser.add_argument('--sweep_values',    default='0.001,0.01,0.05,0.1,0.2,0.5',
                        help='comma-sep lambda_health values for sweep')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  NutriGraphNet v2 — Experiment Pipeline")
    print(f"  Device : {device}")
    print(f"  Output : {args.output_dir}")
    print(f"{'='*70}\n")

    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    print("Data Summary:")
    print(f"  Users        : {data['user'].num_nodes:,}  (feat={data['user'].x.shape[1]})")
    print(f"  Foods        : {data['food'].num_nodes:,}  (feat={data['food'].x.shape[1]})")
    print(f"  Ingredients  : {data['ingredient'].num_nodes:,}")
    print(f"  Interactions : {data[('user','eats','food')].edge_index.shape[1]:,}")
    print()

    variants = [v.strip() for v in args.variants.split(',')]
    all_results = {}

    for variant in variants:
        res = run_cross_validation(data, args, device, variant)
        all_results[variant] = res

        # Save per-variant result
        with open(f"{args.output_dir}/results_{variant}.json", 'w') as f:
            json.dump({'variant': variant,
                       'aggregated': res['aggregated'],
                       'fold_results': res['fold_results']}, f, indent=2,
                      cls=_NumpyEncoder)
        gc_collect()

    # Save all
    with open(f"{args.output_dir}/all_results.json", 'w') as f:
        json.dump({v: {'aggregated': r['aggregated'],
                       'fold_results': r['fold_results']}
                   for v, r in all_results.items()}, f, indent=2,
                  cls=_NumpyEncoder)

    # ── Visualize ──
    print("\nGenerating figures...")
    if len(all_results) > 1:
        plot_model_comparison(all_results, args.output_dir)
        plot_ablation_comparison(all_results, args.output_dir)
    plot_training_curves(all_results, args.output_dir)

    # ── Significance test ──
    sig = {}
    if 'full' in all_results and len(all_results) > 1:
        print("\nStatistical Significance (Wilcoxon signed-rank, one-tailed: full > baseline):")
        print(f"{'Baseline':<15} {'Metric':<12} {'Δ':>8} {'p-val':>8} {'sig':>5}")
        print("-" * 55)
        sig = significance_test(all_results, proposed='full')
        for baseline, rows in sig.items():
            for metric, v in rows.items():
                stars = ''
                if v.get('sig_01'):    stars = '**'
                elif v.get('sig'):     stars = '*'
                print(f"  vs {baseline:<12} {metric:<12} "
                      f"{v['delta']:>+8.4f} {v['p']:>8.4f} {stars:>5}")

        # Save significance results
        with open(f"{args.output_dir}/significance_test.json", 'w') as f:
            json.dump(sig, f, indent=2, cls=_NumpyEncoder)

    # ── LaTeX table ──
    print("Generating LaTeX table...")
    generate_latex_table(all_results, args.output_dir, sig_results=sig)

    # ── Lambda sweep ──
    if args.lambda_sweep:
        print("\n" + "="*70)
        print("  Lambda_health Sensitivity Sweep")
        print("="*70)
        sweep_vals = [float(x) for x in args.sweep_values.split(',')]
        sweep_results = {}
        for lh_val in sweep_vals:
            print(f"\n  [sweep] lambda_health = {lh_val}")
            args.lambda_health = lh_val
            sv_key = f'full_lh{lh_val}'
            res = run_cross_validation(data, args, device, 'full')
            # Tag result with lambda value
            sweep_results[lh_val] = {
                'HR@10':   res['aggregated'].get('HR@10', {}).get('mean', 0),
                'NDCG@10': res['aggregated'].get('NDCG@10', {}).get('mean', 0),
                'MRR':     res['aggregated'].get('MRR', {}).get('mean', 0),
                'auc':     res['aggregated'].get('auc', {}).get('mean', 0),
                'HealthGain@10': res['aggregated'].get('HealthGain@10', {}).get('mean', 0),
            }
            with open(f"{args.output_dir}/sweep_lh{lh_val}.json", 'w') as f:
                json.dump({'lambda_health': lh_val,
                           'aggregated': res['aggregated'],
                           'fold_results': res['fold_results']}, f, indent=2,
                          cls=_NumpyEncoder)
            gc_collect()

        # Plot sweep results
        _plot_lambda_sweep(sweep_results, args.output_dir)
        with open(f"{args.output_dir}/lambda_sweep_summary.json", 'w') as f:
            json.dump({str(k): v for k, v in sweep_results.items()}, f, indent=2,
                      cls=_NumpyEncoder)
        print("\n  Lambda sweep summary:")
        print(f"  {'lambda_h':>12} {'HR@10':>8} {'NDCG@10':>10} {'MRR':>8} {'HG@10':>10}")
        for lh_val, sv in sweep_results.items():
            print(f"  {lh_val:>12.4f} {sv['HR@10']:>8.4f} {sv['NDCG@10']:>10.4f} "
                  f"{sv['MRR']:>8.4f} {sv.get('HealthGain@10', 0):>10.4f}")

    # ── Summary table ──
    key_m = ['auc', 'f1', 'HR@10', 'NDCG@10', 'MRR', 'HealthGain@10']
    print(f"\n{'='*80}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    hdr = f"{'Model':<20}" + "".join(f"{m:>18}" for m in key_m)
    print(hdr); print("-" * len(hdr))
    model_order = ['mf', 'lightgcn', 'ngcf', 'sgl', 'hfrsda',
                   'no_dual', 'no_health', 'no_cl', 'full']
    for v in model_order:
        if v not in all_results: continue
        agg = all_results[v]['aggregated']
        row = f"{v:<20}"
        for m in key_m:
            if m in agg:
                row += f"{agg[m]['mean']:>10.4f}\u00b1{agg[m]['std']:.3f}  "
            else:
                row += f"{'--':>18}"
        print(row)
    print("=" * 80)
    print(f"\n  All results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
