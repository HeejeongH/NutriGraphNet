"""
NutriGraphNet: Health-Aware Food Recommendation System
통합 버전 - HealthAwareGNN 모델 + PrefGNN 학습 파이프라인

Author: Heejeong
Date: 2026-01-01
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch_geometric
from torch_geometric.nn import HeteroConv, GATConv, Linear
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    ndcg_score,
    average_precision_score
)

# ============================================================================
# 1. Health-Aware GAT Encoder (from HealthAwareGNN.py)
# ============================================================================

class HealthAwareGATEncoder(nn.Module):
    """
    Health-Aware GAT Encoder with Residual Connections and Health Attention
    
    Features:
    - 2-layer Heterogeneous GAT
    - Residual connections for better gradient flow
    - Health attention mechanism
    - Layer normalization for stability
    """
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        metadata: Tuple,
        dropout: float = 0.5,
        heads: int = 2
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        
        # First GAT layer
        self.conv1 = HeteroConv({
            edge_type: GATConv(
                in_channels=(-1, -1),
                out_channels=hidden_channels // heads,
                heads=heads,
                dropout=dropout,
                add_self_loops=False,
                concat=True
            )
            for edge_type in metadata[1]
        }, aggr='mean')
        
        # Second GAT layer
        self.conv2 = HeteroConv({
            edge_type: GATConv(
                in_channels=(hidden_channels, hidden_channels),
                out_channels=out_channels,
                heads=1,
                dropout=dropout,
                add_self_loops=False,
                concat=False
            )
            for edge_type in metadata[1]
        }, aggr='mean')
        
        # Health attention mechanism
        self.health_attention = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.LayerNorm(out_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels // 2, 1),
            nn.Sigmoid()
        )
        
        # Batch normalization per node type
        self.batch_norms1 = nn.ModuleDict({
            node_type: nn.BatchNorm1d(hidden_channels)
            for node_type in metadata[0]
        })
        
        self.batch_norms2 = nn.ModuleDict({
            node_type: nn.BatchNorm1d(out_channels)
            for node_type in metadata[0]
        })
        
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        health_scores: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with health awareness
        
        Args:
            x_dict: Node features per type
            edge_index_dict: Edge indices per type
            health_scores: Optional health scores for recipes
            
        Returns:
            Updated node embeddings with health awareness
        """
        # First layer with residual
        x_dict_1 = self.conv1(x_dict, edge_index_dict)
        
        # Apply batch norm and activation
        for node_type, x in x_dict_1.items():
            if node_type in self.batch_norms1:
                x_dict_1[node_type] = F.gelu(self.batch_norms1[node_type](x))
                x_dict_1[node_type] = F.dropout(x_dict_1[node_type], p=self.dropout, training=self.training)
        
        # Second layer with residual
        x_dict_2 = self.conv2(x_dict_1, edge_index_dict)
        
        # Apply batch norm and activation
        for node_type, x in x_dict_2.items():
            if node_type in self.batch_norms2:
                x_dict_2[node_type] = F.gelu(self.batch_norms2[node_type](x))
                x_dict_2[node_type] = F.dropout(x_dict_2[node_type], p=self.dropout, training=self.training)
        
        # Apply health attention if health scores are provided
        if health_scores is not None and 'food' in x_dict_2:
            # Compute attention weights
            attn_weights = self.health_attention(x_dict_2['food'])  # [num_foods, 1]
            
            # Scale food embeddings by health scores
            # Higher health score → stronger embedding
            x_dict_2['food'] = x_dict_2['food'] * (1.0 + attn_weights * health_scores.unsqueeze(-1))
        
        return x_dict_2


# ============================================================================
# 2. Health-Aware Edge Decoder (from HealthAwareGNN.py)
# ============================================================================

class HealthAwareEdgeDecoder(nn.Module):
    """
    Health-Aware Edge Decoder with MLP
    
    Features:
    - 4-layer MLP for better expressiveness
    - Layer normalization for stability
    - GELU activation for smoothness
    """
    def __init__(self, hidden_channels: int, dropout: float = 0.3):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_channels // 4, 1)
        )
        
    def forward(
        self,
        z_dict: Dict[str, torch.Tensor],
        edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode edge probabilities from node embeddings
        
        Args:
            z_dict: Node embeddings per type
            edge_label_index: [2, num_edges] tensor of edge indices
            
        Returns:
            Edge predictions (logits)
        """
        # Get user and food indices
        user_idx = edge_label_index[0]
        food_idx = edge_label_index[1]
        
        # Get embeddings
        user_emb = z_dict['user'][user_idx]
        food_emb = z_dict['food'][food_idx]
        
        # Concatenate and decode
        edge_emb = torch.cat([user_emb, food_emb], dim=-1)
        return self.mlp(edge_emb).squeeze(-1)


# ============================================================================
# 3. Complete Health-Aware Recommender
# ============================================================================

class HealthAwareRecommender(nn.Module):
    """
    Complete Health-Aware Food Recommendation System
    
    Architecture:
    - HealthAwareGATEncoder for node embeddings
    - HealthAwareEdgeDecoder for link prediction
    """
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        metadata: Tuple,
        dropout: float = 0.5,
        heads: int = 2
    ):
        super().__init__()
        
        self.encoder = HealthAwareGATEncoder(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            metadata=metadata,
            dropout=dropout,
            heads=heads
        )
        
        self.decoder = HealthAwareEdgeDecoder(
            hidden_channels=out_channels,
            dropout=dropout
        )
        
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_label_index: torch.Tensor,
        health_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x_dict: Node features
            edge_index_dict: Edge indices
            edge_label_index: Edges to predict
            health_scores: Optional health scores
            
        Returns:
            Edge predictions (logits)
        """
        # Encode
        z_dict = self.encoder(x_dict, edge_index_dict, health_scores)
        
        # Decode
        return self.decoder(z_dict, edge_label_index)


# ============================================================================
# 4. Health-Aware Loss Function (from HealthAwareGNN.py)
# ============================================================================

class HealthAwareLoss(nn.Module):
    """
    Health-Aware Loss Function
    
    Components:
    1. BCE Loss for link prediction
    2. Health Loss for healthy food promotion
    3. Ranking Loss for preference ordering
    """
    def __init__(
        self,
        lambda_health: float = 0.01,
        ranking_weight: float = 0.2,
        margin: float = 1.0,
        pos_weight: float = 1.0
    ):
        super().__init__()
        self.lambda_health = lambda_health
        self.ranking_weight = ranking_weight
        self.margin = margin
        self.pos_weight = pos_weight
        
        # BCEWithLogitsLoss will be created in forward() to ensure correct device
        self.bce = None
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        health_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute combined loss
        
        Args:
            pred: Predictions (logits)
            target: Ground truth labels
            health_scores: Optional health scores
            
        Returns:
            Combined loss
        """
        # Initialize BCE loss on the same device as pred (lazy initialization)
        if self.bce is None:
            self.bce = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([self.pos_weight], device=pred.device)
            )
        
        # Ensure pos_weight is on the same device
        if self.bce.pos_weight.device != pred.device:
            self.bce = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([self.pos_weight], device=pred.device)
            )
        
        # 1. BCE Loss
        bce_loss = self.bce(pred, target)
        
        total_loss = bce_loss
        
        # 2. Health Loss (if health scores are provided)
        if health_scores is not None:
            # Encourage high scores for healthy foods
            # Loss = -mean(health_score * sigmoid(pred))
            health_loss = -(health_scores * torch.sigmoid(pred)).mean()
            total_loss = total_loss + self.lambda_health * health_loss
        
        # 3. Ranking Loss (if we have both positive and negative samples)
        if target.sum() > 0 and (1 - target).sum() > 0:
            pos_mask = target == 1
            neg_mask = target == 0
            
            pos_preds = pred[pos_mask]
            neg_preds = pred[neg_mask]
            
            if len(pos_preds) > 0 and len(neg_preds) > 0:
                # Randomly sample pairs
                num_pairs = min(len(pos_preds), len(neg_preds), 1000)
                pos_sample = pos_preds[:num_pairs]
                neg_sample = neg_preds[:num_pairs]
                
                # Margin ranking loss: pos_pred should be > neg_pred + margin
                ranking_loss = F.margin_ranking_loss(
                    pos_sample,
                    neg_sample,
                    torch.ones_like(pos_sample),
                    margin=self.margin
                )
                
                total_loss = total_loss + self.ranking_weight * ranking_loss
        
        return total_loss


# ============================================================================
# 5. Training and Evaluation Functions (from PrefGNN.py)
# ============================================================================

def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: HeteroData,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch
    
    Returns:
        loss, metrics dict
    """
    model.train()
    
    # Move data to device
    x_dict = {k: v.to(device) for k, v in train_data.x_dict.items()}
    edge_index_dict = {
        k: v.to(device) for k, v in train_data.edge_index_dict.items()
    }
    edge_label_index = train_data[('user', 'eats', 'food')].edge_label_index.to(device)
    edge_label = train_data[('user', 'eats', 'food')].edge_label.to(device)
    
    # Get health scores if available
    health_scores = None
    if hasattr(train_data['food'], 'health_score'):
        health_scores = train_data['food'].health_score.to(device)
    
    # Forward pass
    optimizer.zero_grad()
    pred = model(x_dict, edge_index_dict, edge_label_index, health_scores)
    
    # Compute loss
    loss = criterion(pred, edge_label, health_scores)
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Compute metrics
    with torch.no_grad():
        pred_proba = torch.sigmoid(pred)
        pred_binary = (pred_proba > 0.5).float()
        
        metrics = {
            'accuracy': (pred_binary == edge_label).float().mean().item(),
            'precision': precision_score(
                edge_label.cpu().numpy(),
                pred_binary.cpu().numpy(),
                zero_division=0
            ),
            'recall': recall_score(
                edge_label.cpu().numpy(),
                pred_binary.cpu().numpy(),
                zero_division=0
            ),
            'f1': f1_score(
                edge_label.cpu().numpy(),
                pred_binary.cpu().numpy(),
                zero_division=0
            )
        }
    
    return loss.item(), metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: HeteroData,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate model on validation/test set
    
    Returns:
        loss, metrics dict
    """
    model.eval()
    
    # Move data to device
    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    edge_index_dict = {
        k: v.to(device) for k, v in data.edge_index_dict.items()
    }
    edge_label_index = data[('user', 'eats', 'food')].edge_label_index.to(device)
    edge_label = data[('user', 'eats', 'food')].edge_label.to(device)
    
    # Get health scores if available
    health_scores = None
    if hasattr(data['food'], 'health_score'):
        health_scores = data['food'].health_score.to(device)
    
    # Forward pass
    pred = model(x_dict, edge_index_dict, edge_label_index, health_scores)
    
    # Compute loss
    loss = criterion(pred, edge_label, health_scores)
    
    # Compute metrics
    pred_proba = torch.sigmoid(pred)
    pred_binary = (pred_proba > 0.5).float()
    
    # Move to CPU for sklearn
    y_true = edge_label.cpu().numpy()
    y_pred = pred_binary.cpu().numpy()
    y_score = pred_proba.cpu().numpy()
    
    metrics = {
        'accuracy': (pred_binary == edge_label).float().mean().item(),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5,
        'ap': average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5
    }
    
    return loss.item(), metrics


# ============================================================================
# 6. Cross-Validation Training (from PrefGNN.py)
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience: int = 20, delta: float = 0.0, verbose: bool = True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.best_loss = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


def train_one_fold(
    fold: int,
    train_data: HeteroData,
    val_data: HeteroData,
    test_data: HeteroData,
    args: argparse.Namespace,
    device: torch.device
) -> Dict:
    """
    Train one fold of cross-validation
    
    Returns:
        Dictionary with best metrics and training history
    """
    print(f"\n{'='*80}")
    print(f"🎯 Training Fold {fold + 1}/{args.n_folds}")
    print(f"{'='*80}\n")
    
    # Create model
    model = HealthAwareRecommender(
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        metadata=(train_data.node_types, train_data.edge_types),
        dropout=args.dropout,
        heads=args.heads
    ).to(device)
    
    # Initialize model with a dummy forward pass
    print("🔧 Initializing model...")
    with torch.no_grad():
        # Create a small dummy batch for initialization
        dummy_edge_index = train_data[('user', 'eats', 'food')].edge_label_index[:, :10].to(device)
        _ = model(
            {k: v.to(device) for k, v in train_data.x_dict.items()},
            {k: v.to(device) for k, v in train_data.edge_index_dict.items()},
            dummy_edge_index
        )
    
    # Count parameters (after initialization)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model Parameters: {num_params:,}")
    
    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01  # Minimum LR = 1% of initial LR
    )
    
    # Create criterion
    criterion = HealthAwareLoss(
        lambda_health=args.lambda_health,
        ranking_weight=args.ranking_weight,
        margin=args.margin,
        pos_weight=args.pos_weight
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_auc': [],
        'lr': []
    }
    
    best_val_f1 = 0.0
    best_metrics = None
    
    # Training loop
    for epoch in range(args.epochs):
        # Train
        train_loss, train_metrics = train_epoch(
            model, optimizer, train_data, criterion, device
        )
        
        # Validate
        val_loss, val_metrics = evaluate(model, val_data, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        history['lr'].append(current_lr)
        
        # Print progress
        if (epoch + 1) % args.print_every == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f} | "
                  f"LR: {current_lr:.2e}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            
            # Test on best model
            test_loss, test_metrics = evaluate(model, test_data, criterion, device)
            
            best_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_f1': val_metrics['f1'],
                'val_auc': val_metrics['auc'],
                'test_loss': test_loss,
                'test_f1': test_metrics['f1'],
                'test_auc': test_metrics['auc'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'test_ap': test_metrics['ap']
            }
            
            # Save model
            save_dir = Path(args.output_dir) / f"fold_{fold + 1}"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "best_model.pth")
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\n⏹️  Early stopping at epoch {epoch + 1}")
            break
    
    # Plot training curves
    plot_training_curves(history, fold, args.output_dir)
    
    return {
        'best_metrics': best_metrics,
        'history': history
    }


def plot_training_curves(history: Dict, fold: int, output_dir: str):
    """Plot and save training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1
    axes[0, 1].plot(history['val_f1'], label='Val F1', linewidth=2, color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Score Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 0].plot(history['val_auc'], label='Val AUC', linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].set_title('AUC Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(history['lr'], label='LR', linewidth=2, color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    # Save
    save_path = Path(output_dir) / f"fold_{fold + 1}" / "training_curves.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 7. Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='NutriGraphNet Training')
    
    # Data
    parser.add_argument('--data_path', type=str, 
                       default='data/processed_data/processed_data_GNN_v5.pkl',
                       help='Path to processed data')
    
    # Model
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--out_channels', type=int, default=64)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # Training
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=20)
    
    # Loss
    parser.add_argument('--lambda_health', type=float, default=0.01)
    parser.add_argument('--ranking_weight', type=float, default=0.2)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--pos_weight', type=float, default=1.0)
    
    # Cross-validation
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--test_ratio', type=float, default=0.10)
    parser.add_argument('--neg_sampling_ratio', type=float, default=2.0)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results/final_experiments')
    parser.add_argument('--print_every', type=int, default=10)
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    
    # Load data
    print(f"📂 Loading data from {args.data_path}...")
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ Data loaded successfully!")
    print(f"   Users: {data['user'].num_nodes:,}")
    print(f"   Foods: {data['food'].num_nodes:,}")
    if 'ingredient' in data.node_types:
        print(f"   Ingredients: {data['ingredient'].num_nodes:,}")
    print(f"   Edges: {data[('user', 'eats', 'food')].edge_index.size(1):,}\n")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Cross-validation
    transform = RandomLinkSplit(
        num_val=args.val_ratio,
        num_test=args.test_ratio,
        disjoint_train_ratio=0.1,
        neg_sampling_ratio=args.neg_sampling_ratio,
        edge_types=[('user', 'eats', 'food')],
        rev_edge_types=[('food', 'rev_eats', 'user')]
    )
    
    all_results = []
    
    for fold in range(args.n_folds):
        # Split data
        train_data, val_data, test_data = transform(data)
        
        # Train fold
        result = train_one_fold(
            fold, train_data, val_data, test_data, args, device
        )
        
        all_results.append(result['best_metrics'])
        
        # Print fold results
        print(f"\n📊 Fold {fold + 1} Results:")
        print(f"   Best Epoch: {result['best_metrics']['epoch']}")
        print(f"   Val F1: {result['best_metrics']['val_f1']:.4f}")
        print(f"   Val AUC: {result['best_metrics']['val_auc']:.4f}")
        print(f"   Test F1: {result['best_metrics']['test_f1']:.4f}")
        print(f"   Test AUC: {result['best_metrics']['test_auc']:.4f}")
        print(f"   Test Precision: {result['best_metrics']['test_precision']:.4f}")
        print(f"   Test Recall: {result['best_metrics']['test_recall']:.4f}")
    
    # Compute average results
    print(f"\n{'='*80}")
    print(f"📊 Average Results Across {args.n_folds} Folds")
    print(f"{'='*80}\n")
    
    avg_results = {}
    for key in ['val_f1', 'val_auc', 'test_f1', 'test_auc', 'test_precision', 'test_recall', 'test_ap']:
        values = [r[key] for r in all_results]
        avg_results[key] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
        print(f"{key:20s}: {avg_results[key]['mean']:.4f} ± {avg_results[key]['std']:.4f}")
    
    # Save results
    results_file = Path(args.output_dir) / "cross_validation_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump({
            'all_results': all_results,
            'avg_results': avg_results,
            'args': vars(args)
        }, f)
    
    print(f"\n✅ Results saved to {results_file}")
    print(f"\n🎉 Training completed successfully!\n")


if __name__ == '__main__':
    main()
