"""
NutriGraphNet V2 Training Script - 통합 실험 실행 스크립트
다양한 GNN 모델과 Loss function을 지원하는 범용 훈련 스크립트
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import argparse
import time
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import our modules
import sys
sys.path.append('src')

# 모델 import 시도 (실패 시 해당 모델 비활성화)
try:
    from NutriGraphNet_v2 import (
        NutriGraphNetV2,
        AdaptiveDualObjectiveLoss,
        NegativeSampler,
        FeatureAugmentation
    )
    HAS_V2 = True
except ImportError:
    print("⚠️ Warning: NutriGraphNet_v2 not available")
    HAS_V2 = False

try:
    from HealthAwareGNN import FullHealthAwareGNN
    HAS_HEALTH_GNN = True
except ImportError:
    print("⚠️ Warning: HealthAwareGNN not available")
    HAS_HEALTH_GNN = False

try:
    from training_utils import (
        CosineAnnealingWithWarmRestarts,
        EarlyStopping,
        TrainingMonitor,
        compute_metrics,
        GradientClipper
    )
    HAS_UTILS = True
except ImportError:
    print("⚠️ Warning: training_utils not available, using basic alternatives")
    HAS_UTILS = False

# PyTorch Geometric imports
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv, GraphConv, GCNConv


# ============================================
# Baseline GNN Models
# ============================================

class VanillaGNN(nn.Module):
    """Vanilla GNN with basic graph convolution"""
    def __init__(self, hidden_channels, out_channels, metadata, input_dims):
        super().__init__()
        
        self.input_projections = nn.ModuleDict({
            node_type: nn.Linear(input_dims[node_type], hidden_channels)
            for node_type in metadata[0]
        })
        
        self.conv1 = HeteroConv({
            ('user', 'eats', 'food'): GraphConv(hidden_channels, out_channels),
            ('food', 'rev_eats', 'user'): GraphConv(hidden_channels, out_channels)
        })
        
        self.conv2 = HeteroConv({
            ('user', 'eats', 'food'): GraphConv(out_channels, out_channels),
            ('food', 'rev_eats', 'user'): GraphConv(out_channels, out_channels)
        })
        
        self.decoder = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x_dict, edge_index_dict, edge_label_index, **kwargs):
        current_x = {node_type: self.input_projections[node_type](features)
                    for node_type, features in x_dict.items()}
        
        current_x = self.conv1(current_x, edge_index_dict)
        current_x = {k: F.relu(v) for k, v in current_x.items()}
        
        current_x = self.conv2(current_x, edge_index_dict)
        current_x = {k: F.relu(v) for k, v in current_x.items()}
        
        user_indices, food_indices = edge_label_index
        user_emb = current_x['user'][user_indices]
        food_emb = current_x['food'][food_indices]
        
        combined = torch.cat([user_emb, food_emb], dim=-1)
        return self.decoder(combined).squeeze()


class GraphSAGE_Model(nn.Module):
    """GraphSAGE with neighborhood sampling"""
    def __init__(self, hidden_channels, out_channels, metadata, input_dims):
        super().__init__()
        
        self.input_projections = nn.ModuleDict({
            node_type: nn.Linear(input_dims[node_type], hidden_channels)
            for node_type in metadata[0]
        })
        
        self.conv1 = HeteroConv({
            ('user', 'eats', 'food'): SAGEConv(hidden_channels, out_channels),
            ('food', 'rev_eats', 'user'): SAGEConv(hidden_channels, out_channels)
        })
        
        self.conv2 = HeteroConv({
            ('user', 'eats', 'food'): SAGEConv(out_channels, out_channels),
            ('food', 'rev_eats', 'user'): SAGEConv(out_channels, out_channels)
        })
        
        self.decoder = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x_dict, edge_index_dict, edge_label_index, **kwargs):
        current_x = {node_type: self.input_projections[node_type](features)
                    for node_type, features in x_dict.items()}
        
        current_x = self.conv1(current_x, edge_index_dict)
        current_x = {k: F.relu(v) for k, v in current_x.items()}
        
        current_x = self.conv2(current_x, edge_index_dict)
        current_x = {k: F.relu(v) for k, v in current_x.items()}
        
        user_indices, food_indices = edge_label_index
        user_emb = current_x['user'][user_indices]
        food_emb = current_x['food'][food_indices]
        
        combined = torch.cat([user_emb, food_emb], dim=-1)
        return self.decoder(combined).squeeze()


class GAT_Model(nn.Module):
    """Graph Attention Network"""
    def __init__(self, hidden_channels, out_channels, metadata, input_dims):
        super().__init__()
        
        self.input_projections = nn.ModuleDict({
            node_type: nn.Linear(input_dims[node_type], hidden_channels)
            for node_type in metadata[0]
        })
        
        self.conv = HeteroConv({
            ('user', 'eats', 'food'): GATConv(
                in_channels=(-1, -1), out_channels=out_channels,
                heads=4, concat=False, dropout=0.3, add_self_loops=False, edge_dim=1
            ),
            ('food', 'rev_eats', 'user'): GATConv(
                in_channels=(-1, -1), out_channels=out_channels,
                heads=4, concat=False, dropout=0.3, add_self_loops=False
            )
        })
        
        self.decoder = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
        
        self.batch_norms = nn.ModuleDict({
            node_type: nn.BatchNorm1d(out_channels) for node_type in metadata[0]
        })
    
    def forward(self, x_dict, edge_index_dict, edge_label_index, **kwargs):
        current_x = {node_type: self.input_projections[node_type](features)
                    for node_type, features in x_dict.items()}
        
        current_x = self.conv(current_x, edge_index_dict)
        
        for node_type in current_x.keys():
            if node_type in self.batch_norms:
                current_x[node_type] = F.relu(self.batch_norms[node_type](current_x[node_type]))
        
        user_indices, food_indices = edge_label_index
        user_emb = current_x['user'][user_indices]
        food_emb = current_x['food'][food_indices]
        
        combined = torch.cat([user_emb, food_emb], dim=-1)
        return self.decoder(combined).squeeze()


# ============================================
# Loss Functions
# ============================================

class StandardLoss(nn.Module):
    """Standard Binary Cross Entropy Loss"""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
    
    def forward(self, predictions, targets, **kwargs):
        return self.bce(predictions, targets)


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced data"""
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets, **kwargs):
        ce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class HealthAwareLoss(nn.Module):
    """Health-aware loss with health score regularization"""
    def __init__(self, health_lambda=0.1):
        super().__init__()
        self.health_lambda = health_lambda
        self.bce = nn.BCELoss()
    
    def forward(self, predictions, targets, health_scores=None, **kwargs):
        base_loss = self.bce(predictions, targets)
        
        if health_scores is not None and len(health_scores) > 0:
            health_normalized = torch.sigmoid(health_scores)
            health_reg = -torch.mean(predictions * health_normalized.mean())
            total_loss = base_loss + self.health_lambda * health_reg
        else:
            total_loss = base_loss
        
        return total_loss


# ============================================
# Data Loading & Preprocessing
# ============================================

def load_data(data_path, device):
    """Load data from pickle file"""
    print(f"\n📊 Loading data from {data_path}...")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Move to device
    data = data.to(device)
    
    print(f"✅ Data loaded successfully!")
    print(f"   Users: {data.x_dict['user'].shape[0]:,}")
    print(f"   Foods: {data.x_dict['food'].shape[0]:,}")
    if 'ingredient' in data.x_dict:
        print(f"   Ingredients: {data.x_dict['ingredient'].shape[0]:,}")
    
    return data


def prepare_train_test_split(data, test_ratio=0.2, seed=42):
    """Prepare train/test split"""
    print(f"\n🔀 Preparing train/test split (test ratio: {test_ratio})...")
    
    # Get eats edges
    eats_edge_index = data[('user', 'eats', 'food')].edge_index
    eats_edge_attr = data[('user', 'eats', 'food')].edge_attr
    
    num_edges = eats_edge_index.shape[1]
    
    # Random shuffle
    torch.manual_seed(seed)
    perm = torch.randperm(num_edges)
    
    # Split
    test_size = int(num_edges * test_ratio)
    train_size = num_edges - test_size
    
    train_indices = perm[:train_size]
    test_indices = perm[test_size:]
    
    # Create labels (binary: above median = 1)
    threshold = eats_edge_attr.median()
    labels = (eats_edge_attr > threshold).float()
    
    train_data = {
        'edge_index': eats_edge_index[:, train_indices],
        'labels': labels[train_indices]
    }
    
    test_data = {
        'edge_index': eats_edge_index[:, test_indices],
        'labels': labels[test_indices]
    }
    
    print(f"✅ Split complete!")
    print(f"   Train: {train_size:,} edges")
    print(f"   Test:  {test_size:,} edges")
    print(f"   Threshold: {threshold:.3f}")
    print(f"   Positive ratio: {labels.mean():.3f}")
    
    return train_data, test_data, threshold


# ============================================
# Training & Evaluation
# ============================================

def compute_simple_metrics(predictions, targets):
    """Compute evaluation metrics"""
    predictions_np = predictions.cpu().detach().numpy()
    targets_np = targets.cpu().detach().numpy()
    
    pred_binary = (predictions_np > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(targets_np, pred_binary),
        'precision': precision_score(targets_np, pred_binary, zero_division=0),
        'recall': recall_score(targets_np, pred_binary, zero_division=0),
        'f1': f1_score(targets_np, pred_binary, zero_division=0),
        'auc': roc_auc_score(targets_np, predictions_np) if len(np.unique(targets_np)) > 1 else 0.5
    }
    
    return metrics


def train_epoch(model, data, train_data, optimizer, criterion, device, has_health=False):
    """Train for one epoch"""
    model.train()
    
    # Move data to device
    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    
    train_edge_index = train_data['edge_index'].to(device)
    train_labels = train_data['labels'].to(device)
    
    # Get health edges if available
    health_edge_index = None
    health_scores = None
    if has_health and ('user', 'healthness', 'food') in data.edge_index_dict:
        health_edge_index = data[('user', 'healthness', 'food')].edge_index.to(device)
        health_scores = data[('user', 'healthness', 'food')].edge_attr.to(device)
    
    # Forward pass
    optimizer.zero_grad()
    
    # Model forward (handle different return types)
    output = model(
        x_dict,
        edge_index_dict,
        train_edge_index,
        health_edge_index=health_edge_index,
        health_scores=health_scores,
        training=True
    )
    
    # Handle tuple return (predictions, user_health_prefs)
    if isinstance(output, tuple):
        predictions = output[0]
    else:
        predictions = output
    
    # Extract health scores for train edges (if available)
    batch_health_scores = None
    if health_scores is not None:
        user_indices = train_edge_index[0]
        food_indices = train_edge_index[1]
        
        batch_health_scores = []
        for u_idx, f_idx in zip(user_indices.cpu(), food_indices.cpu()):
            mask = (health_edge_index[0] == u_idx) & (health_edge_index[1] == f_idx)
            if mask.any():
                batch_health_scores.append(health_scores[mask][0])
            else:
                batch_health_scores.append(health_scores.mean())
        
        batch_health_scores = torch.tensor(batch_health_scores, device=device)
    
    # Compute loss
    loss = criterion(predictions, train_labels, health_scores=batch_health_scores)
    
    # Backward
    loss.backward()
    optimizer.step()
    
    # Compute metrics
    with torch.no_grad():
        metrics = compute_simple_metrics(predictions, train_labels)
    
    return loss.item(), metrics


def evaluate(model, data, test_data, criterion, device, has_health=False):
    """Evaluate model"""
    model.eval()
    
    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    
    test_edge_index = test_data['edge_index'].to(device)
    test_labels = test_data['labels'].to(device)
    
    # Get health edges if available
    health_edge_index = None
    health_scores = None
    if has_health and ('user', 'healthness', 'food') in data.edge_index_dict:
        health_edge_index = data[('user', 'healthness', 'food')].edge_index.to(device)
        health_scores = data[('user', 'healthness', 'food')].edge_attr.to(device)
    
    with torch.no_grad():
        # Model forward
        output = model(
            x_dict,
            edge_index_dict,
            test_edge_index,
            health_edge_index=health_edge_index,
            health_scores=health_scores,
            training=False
        )
        
        # Handle tuple return
        if isinstance(output, tuple):
            predictions = output[0]
        else:
            predictions = output
        
        # Extract health scores
        batch_health_scores = None
        if health_scores is not None:
            user_indices = test_edge_index[0]
            food_indices = test_edge_index[1]
            
            batch_health_scores = []
            for u_idx, f_idx in zip(user_indices.cpu(), food_indices.cpu()):
                mask = (health_edge_index[0] == u_idx) & (health_edge_index[1] == f_idx)
                if mask.any():
                    batch_health_scores.append(health_scores[mask][0])
                else:
                    batch_health_scores.append(health_scores.mean())
            
            batch_health_scores = torch.tensor(batch_health_scores, device=device)
        
        # Compute loss
        loss = criterion(predictions, test_labels, health_scores=batch_health_scores)
        
        # Compute metrics
        metrics = compute_simple_metrics(predictions, test_labels)
    
    return loss.item(), metrics


# ============================================
# Model Factory
# ============================================

def create_model(model_name, hidden_channels, out_channels, metadata, input_dims, device):
    """Create model by name"""
    
    model_dict = {
        'vanilla': VanillaGNN,
        'graphsage': GraphSAGE_Model,
        'gat': GAT_Model,
    }
    
    # Add conditional models
    if HAS_V2:
        model_dict['nutrigraphnet_v2'] = NutriGraphNetV2
    
    if HAS_HEALTH_GNN:
        model_dict['health_gnn'] = FullHealthAwareGNN
    
    if model_name not in model_dict:
        available = ', '.join(model_dict.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    model_class = model_dict[model_name]
    
    # Create model with appropriate arguments
    if model_name == 'nutrigraphnet_v2':
        model = model_class(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            metadata=metadata,
            dropout=0.3,
            num_layers=3,
            device=device
        )
    else:
        model = model_class(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            metadata=metadata,
            input_dims=input_dims
        )
    
    return model.to(device)


def create_loss_function(loss_name, **kwargs):
    """Create loss function by name"""
    
    loss_dict = {
        'standard': StandardLoss,
        'focal': FocalLoss,
        'health': HealthAwareLoss,
    }
    
    # Add conditional loss functions
    if HAS_V2:
        loss_dict['adaptive'] = AdaptiveDualObjectiveLoss
    
    if loss_name not in loss_dict:
        available = ', '.join(loss_dict.keys())
        raise ValueError(f"Unknown loss: {loss_name}. Available: {available}")
    
    loss_class = loss_dict[loss_name]
    
    # Create loss with appropriate arguments
    if loss_name == 'focal':
        return loss_class(alpha=kwargs.get('alpha', 1), gamma=kwargs.get('gamma', 2))
    elif loss_name == 'health':
        return loss_class(health_lambda=kwargs.get('health_lambda', 0.1))
    elif loss_name == 'adaptive' and HAS_V2:
        return loss_class(
            lambda_health_init=kwargs.get('lambda_health_init', 0.01),
            lambda_health_max=kwargs.get('lambda_health_max', 0.1),
            focal_gamma=kwargs.get('focal_gamma', 2.0),
            temp=kwargs.get('temperature', 2.0)
        )
    else:
        return loss_class()


# ============================================
# Main Training Loop
# ============================================

def main(args):
    """Main training function"""
    print("="*80)
    print(f"🚀 NutriGraphNet Training - Model: {args.model}")
    print("="*80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    data = load_data(args.data_path, device)
    
    # Prepare splits
    train_data, test_data, threshold = prepare_train_test_split(
        data, test_ratio=args.test_ratio, seed=args.seed
    )
    
    # Model metadata
    metadata = (list(data.x_dict.keys()), list(data.edge_index_dict.keys()))
    input_dims = {k: v.shape[1] for k, v in data.x_dict.items()}
    
    # Create model
    print(f"\n🏗️ Creating model: {args.model}...")
    print(f"   Hidden channels: {args.hidden_channels}")
    print(f"   Output channels: {args.out_channels}")
    
    model = create_model(
        args.model, args.hidden_channels, args.out_channels, 
        metadata, input_dims, device
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {num_params:,}")
    
    # Loss function
    print(f"\n📊 Loss function: {args.loss}")
    criterion = create_loss_function(
        args.loss,
        health_lambda=args.health_lambda,
        alpha=1, gamma=2,
        lambda_health_init=args.lambda_health_init,
        lambda_health_max=args.lambda_health_max,
        focal_gamma=args.focal_gamma,
        temperature=args.temperature
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    print(f"\n📚 Training configuration:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Weight decay: {args.weight_decay}")
    
    # Check if model uses health information
    has_health = 'health' in args.model or args.loss == 'health' or args.loss == 'adaptive'
    
    # Training loop
    print(f"\n{'='*80}")
    print("🎯 Starting training...")
    print(f"{'='*80}\n")
    
    best_f1 = 0
    best_metrics = None
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Update adaptive lambda if using adaptive loss
        if hasattr(criterion, 'update_epoch'):
            criterion.update_epoch(epoch, args.epochs)
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, data, train_data, optimizer, criterion, device, has_health
        )
        
        # Evaluate
        val_loss, val_metrics = evaluate(model, data, test_data, criterion, device, has_health)
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        if (epoch + 1) % args.print_every == 0:
            lambda_str = f" | λ_h: {criterion.lambda_health_current:.3f}" if hasattr(criterion, 'lambda_health_current') else ""
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f} | "
                  f"LR: {current_lr:.2e}{lambda_str} | "
                  f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_metrics = val_metrics
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_f1,
                'val_metrics': val_metrics,
                'args': vars(args)
            }, args.save_path)
            print(f"  💾 Saved best model (F1: {best_f1:.4f})")
    
    total_time = time.time() - start_time
    
    # Final evaluation
    print(f"\n{'='*80}")
    print("📊 Training Complete!")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best validation F1: {best_f1:.4f}")
    
    # Load best model and final evaluation
    print(f"\n🔍 Final evaluation with best model...")
    checkpoint = torch.load(args.save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_loss, final_metrics = evaluate(model, data, test_data, criterion, device, has_health)
    
    print(f"\n{'='*80}")
    print("🏆 Final Test Results")
    print(f"{'='*80}")
    print(f"Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall:    {final_metrics['recall']:.4f}")
    print(f"F1 Score:  {final_metrics['f1']:.4f}")
    print(f"AUC:       {final_metrics['auc']:.4f}")
    print(f"{'='*80}\n")
    
    # Save results to JSON
    results = {
        'model': args.model,
        'loss': args.loss,
        'final_metrics': final_metrics,
        'best_f1': best_f1,
        'total_time': total_time,
        'args': vars(args)
    }
    
    results_path = args.result_file
    
    # 디렉토리가 없으면 생성
    results_dir = Path(results_path).parent
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {results_path}\n")
    
    return final_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GNN models for food recommendation')
    
    # Data
    parser.add_argument('--data_path', type=str, 
                       default='data/processed_data/processed_data_GNN_cpu.pkl',
                       help='Path to processed data')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Model
    parser.add_argument('--model', type=str, default='vanilla',
                       choices=['vanilla', 'graphsage', 'gat', 'nutrigraphnet_v2', 'health_gnn'],
                       help='Model architecture')
    parser.add_argument('--hidden_channels', type=int, default=128,
                       help='Hidden channels')
    parser.add_argument('--out_channels', type=int, default=64,
                       help='Output channels')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.02,
                       help='Weight decay')
    
    # Loss
    parser.add_argument('--loss', type=str, default='standard',
                       choices=['standard', 'focal', 'health', 'adaptive'],
                       help='Loss function')
    parser.add_argument('--health_lambda', type=float, default=0.1,
                       help='Health loss weight (for health loss)')
    parser.add_argument('--lambda_health_init', type=float, default=0.01,
                       help='Initial health loss weight (for adaptive loss)')
    parser.add_argument('--lambda_health_max', type=float, default=0.1,
                       help='Maximum health loss weight (for adaptive loss)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma')
    parser.add_argument('--temperature', type=float, default=2.0,
                       help='Temperature scaling')
    
    # Misc
    parser.add_argument('--save_path', type=str, default='best_model.pth',
                       help='Model save path')
    parser.add_argument('--result_file', type=str, default='best_model_results.json',
                       help='Result JSON file path')
    parser.add_argument('--print_every', type=int, default=5,
                       help='Print every N epochs')
    
    args = parser.parse_args()
    
    # Run training
    final_metrics = main(args)
