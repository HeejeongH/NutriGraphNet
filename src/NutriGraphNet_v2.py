"""
NutriGraphNet V2: 성능 개선 버전

개선사항:
1. Multi-head Attention (1 -> 4 heads)
2. Deeper Architecture (2 -> 3 layers)
3. Skip Connections (ResNet-style)
4. Layer-wise Attention Aggregation
5. Adaptive Health Loss Weight
6. Negative Sampling for Better Learning
7. Feature Augmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear
import numpy as np


class ImprovedGATEncoder(torch.nn.Module):
    """
    개선된 GAT Encoder
    
    Improvements:
    - 3 layers (deeper network)
    - 4 attention heads (richer representations)
    - Skip connections (better gradient flow)
    - Layer-wise feature aggregation
    """
    
    def __init__(self, hidden_channels, out_channels, metadata, dropout=0.3, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.skips = nn.ModuleList()
        
        # Input projection layers
        self.input_proj = nn.ModuleDict({
            node_type: Linear(-1, hidden_channels)
            for node_type in metadata[0]
        })
        
        # Adaptive health preference network (더 복잡한 구조)
        self.health_preference_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )
        
        # Build layers
        for i in range(num_layers):
            in_channels = hidden_channels
            out_dim = out_channels if i == num_layers - 1 else hidden_channels
            num_heads = 4 if i < num_layers - 1 else 1  # Multi-head for early layers
            
            # Heterogeneous convolution
            conv = HeteroConv({
                ('user', 'healthness', 'food'): GATConv(
                    in_channels=(in_channels, in_channels),
                    out_channels=out_dim,
                    heads=num_heads,
                    concat=False,  # Average multi-head outputs
                    dropout=dropout,
                    add_self_loops=False,
                    edge_dim=1
                ),
                **{edge_type: GATConv(
                    in_channels=(in_channels, in_channels),
                    out_channels=out_dim,
                    heads=num_heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=False
                ) for edge_type in metadata[1] 
                if edge_type not in [('user', 'healthness', 'food')]}
            })
            self.convs.append(conv)
            
            # Batch normalization
            self.bns.append(nn.ModuleDict({
                node_type: nn.BatchNorm1d(out_dim)
                for node_type in metadata[0]
            }))
            
            # Skip connections
            if i < num_layers - 1:
                self.skips.append(nn.ModuleDict({
                    node_type: Linear(in_channels, out_dim) if in_channels != out_dim else nn.Identity()
                    for node_type in metadata[0]
                }))
        
        self.dropout = nn.Dropout(dropout)
        
        # Layer-wise attention weights (학습 가능한 레이어 중요도)
        self.layer_attention = nn.Parameter(torch.ones(num_layers))
    
    def forward(self, x_dict, edge_index_dict, health_edge_index=None, health_scores=None):
        # Input projection
        x_dict = {
            node_type: self.input_proj[node_type](x)
            for node_type, x in x_dict.items()
        }
        
        layer_outputs = []
        
        # Multi-layer propagation with skip connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_prev = {k: v.clone() for k, v in x_dict.items()}
            
            # Graph convolution
            x_dict = conv(x_dict, edge_index_dict)
            
            # Batch norm + activation
            x_dict = {key: F.elu(bn[key](x)) for key, x in x_dict.items()}
            
            # Skip connection (for non-final layers)
            if i < len(self.skips):
                x_dict = {
                    key: x + self.skips[i][key](x_prev[key])
                    for key, x in x_dict.items()
                }
            
            # Dropout
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
            
            layer_outputs.append(x_dict)
        
        # Layer-wise attention aggregation
        layer_weights = F.softmax(self.layer_attention, dim=0)
        aggregated = {}
        for node_type in x_dict.keys():
            stacked = torch.stack([output[node_type] for output in layer_outputs])
            aggregated[node_type] = torch.sum(
                stacked * layer_weights.view(-1, 1, 1), 
                dim=0
            )
        
        x_dict = aggregated
        
        # Health attention (최종 레이어 이후)
        user_health_preferences = None
        if health_edge_index is not None and health_scores is not None:
            user_indices = health_edge_index[0]
            food_indices = health_edge_index[1]
            
            # User health preferences (adaptive)
            user_health_preferences = self.health_preference_layer(x_dict['user'])
            user_specific_weights = user_health_preferences[user_indices].squeeze()
            
            # Adjusted health scores
            adjusted_health_scores = user_specific_weights * health_scores
            
            # Food embedding update (더 세밀한 조정)
            food_dim = x_dict['food'].size(1)
            health_weighted_values = adjusted_health_scores.unsqueeze(-1).expand(-1, food_dim)
            
            # Aggregation with normalization
            food_updates = torch.zeros_like(x_dict['food'])
            food_counts = torch.zeros(x_dict['food'].size(0), device=x_dict['food'].device)
            
            food_updates.scatter_add_(
                0, 
                food_indices.unsqueeze(-1).expand(-1, food_dim), 
                health_weighted_values
            )
            food_counts.scatter_add_(
                0, 
                food_indices, 
                torch.ones_like(food_indices, dtype=torch.float)
            )
            
            # Normalize by count
            food_updates = food_updates / (food_counts.unsqueeze(-1) + 1e-8)
            
            # Dynamic health influence (학습 가능)
            health_scale = nn.Parameter(torch.tensor(0.1))
            x_dict['food'] = x_dict['food'] + health_scale * food_updates
        
        return x_dict, user_health_preferences


class ImprovedDecoder(torch.nn.Module):
    """
    개선된 Decoder
    
    Improvements:
    - Deeper MLP (2 -> 3 layers)
    - Residual connection
    - Feature interaction layer
    """
    
    def __init__(self, hidden_channels, dropout=0.3, inference_bias=0.05):
        super().__init__()
        
        # Feature interaction
        self.interaction = nn.Bilinear(hidden_channels, hidden_channels, hidden_channels)
        
        # Deep MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels * 2),  # concat + interaction
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_channels, 1)
        )
        
        self.inference_bias = inference_bias
    
    def forward(self, z_dict, edge_label_index, training=True):
        row, col = edge_label_index
        user_emb = z_dict['user'][row]
        food_emb = z_dict['food'][col]
        
        # Feature interaction
        interaction = self.interaction(user_emb, food_emb)
        
        # Concatenate all features
        z = torch.cat([user_emb, food_emb, interaction], dim=-1)
        
        # MLP
        z = self.mlp(z)
        
        # Inference bias (더 작게 조정)
        if training:
            return torch.sigmoid(z.view(-1))
        else:
            return torch.sigmoid((z + self.inference_bias).view(-1))


class AdaptiveDualObjectiveLoss(nn.Module):
    """
    적응형 Dual-Objective Loss
    
    Improvements:
    - Adaptive lambda_health (epoch에 따라 조정)
    - Focal loss for imbalanced data
    - Temperature scaling for health scores
    """
    
    def __init__(self, lambda_health_init=0.01, lambda_health_max=0.1, 
                 focal_gamma=2.0, temp=2.0):
        super().__init__()
        self.lambda_health_current = lambda_health_init
        self.lambda_health_init = lambda_health_init
        self.lambda_health_max = lambda_health_max
        self.focal_gamma = focal_gamma
        self.temp = temp
        self.epoch = 0
    
    def update_epoch(self, epoch, total_epochs):
        """Epoch에 따라 health loss 가중치 조정"""
        # Warm-up: 초반에는 preference만 학습, 후반에 건강 고려
        progress = epoch / total_epochs
        self.lambda_health_current = self.lambda_health_init + \
            (self.lambda_health_max - self.lambda_health_init) * progress
        self.epoch = epoch
    
    def focal_loss(self, predictions, targets):
        """Focal Loss for imbalanced data"""
        bce = F.binary_cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = ((1 - pt) ** self.focal_gamma) * bce
        return focal_loss.mean()
    
    def forward(self, predictions, targets, health_scores, user_health_preferences=None):
        # 1. Focal Loss (better for imbalanced data)
        preference_loss = self.focal_loss(predictions, targets)
        
        # 2. Temperature-scaled Health Loss
        # 온도를 높여서 극단적인 건강 점수의 영향 완화
        scaled_health = torch.sigmoid((health_scores - 0.5) * self.temp)
        health_loss = -torch.mean(predictions * scaled_health)
        
        # 3. User preference weighted (optional)
        if user_health_preferences is not None:
            user_weights = user_health_preferences.squeeze()
            weighted_health_loss = -torch.mean(predictions * scaled_health * user_weights)
            health_loss = 0.7 * health_loss + 0.3 * weighted_health_loss
        
        # 4. Adaptive combination
        total_loss = preference_loss + self.lambda_health_current * health_loss
        
        return total_loss
    
    def get_components(self, predictions, targets, health_scores):
        """Loss components for monitoring"""
        preference_loss = self.focal_loss(predictions, targets)
        scaled_health = torch.sigmoid((health_scores - 0.5) * self.temp)
        health_loss = -torch.mean(predictions * scaled_health)
        
        return {
            'preference_loss': preference_loss.item(),
            'health_loss': health_loss.item(),
            'lambda_health': self.lambda_health_current,
            'total_loss': (preference_loss + self.lambda_health_current * health_loss).item()
        }


class NutriGraphNetV2(torch.nn.Module):
    """
    NutriGraphNet Version 2 - Improved Performance
    """
    
    def __init__(self, hidden_channels, out_channels, metadata, 
                 dropout=0.3, num_layers=3, device=None):
        super().__init__()
        self.encoder = ImprovedGATEncoder(
            hidden_channels, out_channels, metadata, dropout, num_layers
        )
        self.decoder = ImprovedDecoder(out_channels, dropout, inference_bias=0.05)
        self.device = device
    
    def forward(self, x_dict, edge_index_dict, edge_label_index, 
                health_edge_index=None, health_scores=None, training=True):
        # Type conversion
        edge_label_index = edge_label_index.long()
        edge_index_dict = {k: v.long() for k, v in edge_index_dict.items()}
        
        # Encoding
        z_dict, user_health_preferences = self.encoder(
            x_dict, edge_index_dict, health_edge_index, health_scores
        )
        
        # Decoding
        predictions = self.decoder(z_dict, edge_label_index, training=training)
        
        return predictions, user_health_preferences


# ============================================
# 추가 개선 기법들
# ============================================

class NegativeSampler:
    """
    Negative Sampling for Better Contrastive Learning
    """
    
    def __init__(self, num_negatives=5):
        self.num_negatives = num_negatives
    
    def sample(self, pos_edge_index, num_nodes):
        """Generate negative samples"""
        num_pos = pos_edge_index.size(1)
        num_users, num_foods = num_nodes
        
        # Random negative sampling
        neg_users = torch.randint(0, num_users, (num_pos * self.num_negatives,))
        neg_foods = torch.randint(0, num_foods, (num_pos * self.num_negatives,))
        
        neg_edge_index = torch.stack([neg_users, neg_foods])
        
        return neg_edge_index


class FeatureAugmentation:
    """
    Feature Augmentation for Better Generalization
    """
    
    def __init__(self, noise_std=0.1, dropout_prob=0.1):
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
    
    def augment(self, x_dict, training=True):
        """Apply feature augmentation"""
        if not training:
            return x_dict
        
        augmented = {}
        for node_type, features in x_dict.items():
            # Gaussian noise
            noise = torch.randn_like(features) * self.noise_std
            
            # Feature dropout
            mask = torch.bernoulli(
                torch.ones_like(features) * (1 - self.dropout_prob)
            )
            
            augmented[node_type] = (features + noise) * mask
        
        return augmented


# ============================================
# 사용 예제
# ============================================

def create_improved_model(metadata, device):
    """Create improved model with all enhancements"""
    
    model = NutriGraphNetV2(
        hidden_channels=256,  # Increased capacity
        out_channels=128,     # Increased capacity
        metadata=metadata,
        dropout=0.3,
        num_layers=3,         # Deeper network
        device=device
    ).to(device)
    
    # Adaptive loss
    criterion = AdaptiveDualObjectiveLoss(
        lambda_health_init=0.01,   # Start small
        lambda_health_max=0.1,     # Gradually increase
        focal_gamma=2.0,
        temp=2.0
    )
    
    # Negative sampler
    neg_sampler = NegativeSampler(num_negatives=5)
    
    # Feature augmentation
    feature_aug = FeatureAugmentation(noise_std=0.1, dropout_prob=0.1)
    
    return model, criterion, neg_sampler, feature_aug
