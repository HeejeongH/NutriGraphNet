"""
NutriGraphNet V2 Training Script
개선된 V2 모델 훈련 실행 스크립트
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import argparse
import time
from pathlib import Path

# Import our modules
import sys
sys.path.append('src')

from NutriGraphNet_v2 import (
    NutriGraphNetV2,
    AdaptiveDualObjectiveLoss,
    NegativeSampler,
    FeatureAugmentation
)
from training_utils import (
    CosineAnnealingWithWarmRestarts,
    EarlyStopping,
    TrainingMonitor,
    compute_metrics,
    GradientClipper
)
from health_score_calculator import (
    PersonalizedHealthScoreCalculator,
    precompute_health_scores_for_dataset
)
# For pickle compatibility
from simple_hetero_data import SimpleHeteroData, SimpleEdgeData


def load_data(data_path):
    """데이터 로드"""
    print(f"\n📊 Loading data from {data_path}...")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ Data loaded successfully!")
    print(f"   Users: {data.x_dict['user'].shape[0]:,}")
    print(f"   Foods: {data.x_dict['food'].shape[0]:,}")
    print(f"   Ingredients: {data.x_dict['ingredient'].shape[0]:,}")
    
    return data


def prepare_train_test_split(data, test_ratio=0.2, seed=42):
    """Train/Test split"""
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


def train_epoch(model, data, train_data, optimizer, criterion, scheduler, 
                grad_clipper, feature_aug, device):
    """한 에포크 훈련"""
    model.train()
    
    # Move to device
    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    
    # Feature augmentation
    x_dict = feature_aug.augment(x_dict, training=True)
    
    train_edge_index = train_data['edge_index'].to(device)
    train_labels = train_data['labels'].to(device)
    
    # Health edges
    health_edge_index = data[('user', 'healthness', 'food')].edge_index.to(device)
    health_scores = data[('user', 'healthness', 'food')].edge_attr.to(device)
    
    # Forward pass
    optimizer.zero_grad()
    
    predictions, user_health_prefs = model(
        x_dict,
        edge_index_dict,
        train_edge_index,
        health_edge_index=health_edge_index,
        health_scores=health_scores,
        training=True
    )
    
    # Extract health scores for train edges
    user_indices = train_edge_index[0]
    food_indices = train_edge_index[1]
    
    # Match health scores
    batch_health_scores = []
    for u_idx, f_idx in zip(user_indices.cpu(), food_indices.cpu()):
        mask = (health_edge_index[0] == u_idx) & (health_edge_index[1] == f_idx)
        if mask.any():
            batch_health_scores.append(health_scores[mask][0])
        else:
            batch_health_scores.append(health_scores.mean())
    
    batch_health_scores = torch.tensor(batch_health_scores, device=device)
    
    # Loss
    loss = criterion(predictions, train_labels, batch_health_scores, user_health_prefs)
    
    # Backward
    loss.backward()
    grad_norm = grad_clipper.clip(model)
    optimizer.step()
    scheduler.step()
    
    # Metrics
    with torch.no_grad():
        metrics = compute_metrics(predictions, train_labels)
    
    return loss.item(), metrics, grad_norm


def evaluate(model, data, test_data, criterion, device):
    """평가"""
    model.eval()
    
    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    
    test_edge_index = test_data['edge_index'].to(device)
    test_labels = test_data['labels'].to(device)
    
    health_edge_index = data[('user', 'healthness', 'food')].edge_index.to(device)
    health_scores = data[('user', 'healthness', 'food')].edge_attr.to(device)
    
    with torch.no_grad():
        predictions, user_health_prefs = model(
            x_dict,
            edge_index_dict,
            test_edge_index,
            health_edge_index=health_edge_index,
            health_scores=health_scores,
            training=False  # Inference mode
        )
        
        # Extract health scores
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
        
        # Loss
        loss = criterion(predictions, test_labels, batch_health_scores, user_health_prefs)
        
        # Metrics
        metrics = compute_metrics(predictions, test_labels)
    
    return loss.item(), metrics


def main(args):
    """메인 훈련 루프"""
    print("="*80)
    print("🚀 NutriGraphNet V2 Training")
    print("="*80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    data = load_data(args.data_path)
    data = data.to(device)
    
    # Prepare splits
    train_data, test_data, threshold = prepare_train_test_split(
        data, test_ratio=args.test_ratio, seed=args.seed
    )
    
    # Model metadata
    metadata = (list(data.x_dict.keys()), list(data.edge_index_dict.keys()))
    
    # Create model
    print(f"\n🏗️ Creating model...")
    print(f"   Hidden channels: {args.hidden_channels}")
    print(f"   Output channels: {args.out_channels}")
    print(f"   Num layers: {args.num_layers}")
    print(f"   Dropout: {args.dropout}")
    
    model = NutriGraphNetV2(
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        metadata=metadata,
        dropout=args.dropout,
        num_layers=args.num_layers,
        device=device
    ).to(device)
    
    # Initialize lazy modules with dummy forward pass
    print("   Initializing model parameters...")
    try:
        with torch.no_grad():
            # Dummy forward pass to initialize LazyLinear
            dummy_edge_index = train_data['edge_index'][:, :10].to(device)
            _ = model(
                {k: v.to(device) for k, v in data.x_dict.items()},
                {k: v.to(device) for k, v in data.edge_index_dict.items()},
                dummy_edge_index,
                training=False
            )
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {num_params:,}")
    except Exception as e:
        print(f"   Warning: Could not compute parameter count: {e}")
        print(f"   Continuing with training...")
    
    # Loss function
    criterion = AdaptiveDualObjectiveLoss(
        lambda_health_init=args.lambda_health_init,
        lambda_health_max=args.lambda_health_max,
        focal_gamma=args.focal_gamma,
        temp=args.temperature
    )
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer,
        T_0=args.T_0,
        T_mult=args.T_mult,
        eta_min=args.eta_min,
        eta_max=args.lr
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        mode='max',
        verbose=True
    )
    
    # Training utilities
    monitor = TrainingMonitor()
    grad_clipper = GradientClipper(max_norm=args.max_grad_norm)
    feature_aug = FeatureAugmentation(
        noise_std=args.noise_std,
        dropout_prob=args.feature_dropout
    )
    
    print(f"\n📚 Training configuration:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Weight decay: {args.weight_decay}")
    print(f"   Lambda health: {args.lambda_health_init} -> {args.lambda_health_max}")
    print(f"   Focal gamma: {args.focal_gamma}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Early stopping patience: {args.patience}")
    
    # Training loop
    print(f"\n{'='*80}")
    print("🎯 Starting training...")
    print(f"{'='*80}\n")
    
    best_f1 = 0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Update adaptive lambda
        criterion.update_epoch(epoch, args.epochs)
        
        # Train
        train_loss, train_metrics, grad_norm = train_epoch(
            model, data, train_data, optimizer, criterion,
            scheduler, grad_clipper, feature_aug, device
        )
        
        # Evaluate
        val_loss, val_metrics = evaluate(model, data, test_data, criterion, device)
        
        # Learning rate
        current_lr = scheduler.get_last_lr()[0]
        
        # Log
        monitor.log_epoch(
            epoch,
            train_loss,
            val_loss,
            train_metrics,
            val_metrics,
            current_lr
        )
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        if (epoch + 1) % args.print_every == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"λ_h: {criterion.lambda_health_current:.3f} | "
                  f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_f1,
                'val_metrics': val_metrics
            }, args.save_path)
            print(f"  💾 Saved best model (F1: {best_f1:.4f})")
        
        # Early stopping
        if early_stopping(val_metrics['f1'], epoch, model):
            print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    
    # Training summary
    print(f"\n{'='*80}")
    print("📊 Training Complete!")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Best epoch: {early_stopping.best_epoch + 1}")
    
    monitor.print_summary()
    
    # Load best model and final evaluation
    print(f"\n🔍 Final evaluation with best model...")
    checkpoint = torch.load(args.save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_loss, final_metrics = evaluate(model, data, test_data, criterion, device)
    
    print(f"\n{'='*80}")
    print("🏆 Final Test Results")
    print(f"{'='*80}")
    print(f"Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall:    {final_metrics['recall']:.4f}")
    print(f"F1 Score:  {final_metrics['f1']:.4f}")
    print(f"AUC:       {final_metrics['auc']:.4f}")
    print(f"{'='*80}\n")
    
    return final_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NutriGraphNet V2')
    
    # Data
    parser.add_argument('--data_path', type=str, 
                       default='../data/processed_data/processed_data_GNN.pkl',
                       help='Path to processed data')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Model
    parser.add_argument('--hidden_channels', type=int, default=256,
                       help='Hidden channels')
    parser.add_argument('--out_channels', type=int, default=128,
                       help='Output channels')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.02,
                       help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Max gradient norm')
    
    # Loss
    parser.add_argument('--lambda_health_init', type=float, default=0.01,
                       help='Initial health loss weight')
    parser.add_argument('--lambda_health_max', type=float, default=0.1,
                       help='Maximum health loss weight')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma')
    parser.add_argument('--temperature', type=float, default=2.0,
                       help='Temperature scaling')
    
    # Scheduler
    parser.add_argument('--T_0', type=int, default=10,
                       help='Cosine annealing T_0')
    parser.add_argument('--T_mult', type=int, default=2,
                       help='Cosine annealing T_mult')
    parser.add_argument('--eta_min', type=float, default=1e-6,
                       help='Minimum learning rate')
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.0001,
                       help='Minimum improvement')
    
    # Augmentation
    parser.add_argument('--noise_std', type=float, default=0.1,
                       help='Feature noise std')
    parser.add_argument('--feature_dropout', type=float, default=0.1,
                       help='Feature dropout probability')
    
    # Misc
    parser.add_argument('--save_path', type=str, default='best_model_v2.pth',
                       help='Model save path')
    parser.add_argument('--print_every', type=int, default=5,
                       help='Print every N epochs')
    
    args = parser.parse_args()
    
    # Run training
    final_metrics = main(args)
