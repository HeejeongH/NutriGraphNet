"""
NutriGraphNet: Health-Aware Graph Neural Network for Food Recommendation

논문 아키텍처를 완전히 구현한 버전:
1. Heterogeneous Graph Structure
2. Health Attention Mechanism  
3. Dual-Objective Loss Function
4. Edge Decoder with Inference Bias
5. Cosine Annealing with Warm Restarts
6. Early Stopping with Adaptive Patience
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv


class NutriGraphNetEncoder(torch.nn.Module):
    """
    논문의 Health-aware Graph Attention Encoder
    
    Features:
    - 2-layer Heterogeneous GAT
    - Health attention mechanism
    - Batch normalization & Dropout
    """
    
    def __init__(self, hidden_channels, out_channels, metadata, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # Health preference network (개인화된 건강 선호도 학습)
        self.health_preference_layer = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.ReLU(),
            nn.Linear(out_channels // 2, 1),
            nn.Sigmoid()  # 0 ~ 1 사이의 건강 선호도 가중치
        )
        
        # Layer 1: Input -> Hidden
        conv1 = HeteroConv({
            ('user', 'healthness', 'food'): GATConv(
                in_channels=(-1, -1),
                out_channels=hidden_channels,
                heads=1,
                dropout=dropout,
                add_self_loops=False,
                edge_dim=1  # Health score as edge attribute
            ),
            **{edge_type: GATConv(
                in_channels=(-1, -1),
                out_channels=hidden_channels,
                heads=1,
                dropout=dropout,
                add_self_loops=False
            ) for edge_type in metadata[1] 
            if edge_type not in [('user', 'healthness', 'food')]}
        })
        self.convs.append(conv1)
        self.bns.append(nn.ModuleDict({
            node_type: nn.BatchNorm1d(hidden_channels)
            for node_type in metadata[0]
        }))
        
        # Layer 2: Hidden -> Output
        conv2 = HeteroConv({
            ('user', 'healthness', 'food'): GATConv(
                in_channels=(hidden_channels, hidden_channels),
                out_channels=out_channels,
                heads=1,
                dropout=dropout,
                add_self_loops=False,
                edge_dim=1
            ),
            **{edge_type: GATConv(
                in_channels=(hidden_channels, hidden_channels),
                out_channels=out_channels,
                heads=1,
                dropout=dropout,
                add_self_loops=False
            ) for edge_type in metadata[1] 
            if edge_type not in [('user', 'healthness', 'food')]}
        })
        self.convs.append(conv2)
        self.bns.append(nn.ModuleDict({
            node_type: nn.BatchNorm1d(out_channels)
            for node_type in metadata[0]
        }))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_dict, edge_index_dict, health_edge_index=None, health_scores=None):
        """
        Forward pass with health attention mechanism
        
        Args:
            x_dict: Node features dictionary
            edge_index_dict: Edge indices dictionary
            health_edge_index: Health edges (user-food pairs)
            health_scores: Personalized health scores
            
        Returns:
            tuple: (encoded node features, user health preferences)
        """
        # Graph convolution layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.elu(bn[key](x)) for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
            
            # Health attention integration (논문의 핵심)
            if i == len(self.convs) - 1 and health_edge_index is not None and health_scores is not None:
                user_indices = health_edge_index[0]
                food_indices = health_edge_index[1]
                
                # 사용자별 건강 선호도 가중치 계산
                user_health_preferences = self.health_preference_layer(x_dict['user'])
                user_specific_weights = user_health_preferences[user_indices]
                
                # 개인화된 건강 점수 조정
                adjusted_health_scores = user_specific_weights.squeeze() * health_scores
                
                # Food embedding 업데이트
                food_dim = x_dict['food'].size(1)
                health_weighted_values = adjusted_health_scores.unsqueeze(-1).expand(-1, food_dim)
                
                food_updates = torch.zeros_like(x_dict['food'])
                food_updates.scatter_add_(
                    0, 
                    food_indices.unsqueeze(-1).expand(-1, food_dim), 
                    health_weighted_values
                )
                
                # 논문의 식: x_food' = x_food + 0.1 × health_update
                x_dict['food'] = x_dict['food'] + 0.1 * food_updates
                
                return x_dict, user_health_preferences
        
        return x_dict, None


class NutriGraphNetDecoder(torch.nn.Module):
    """
    논문의 Edge Decoder
    
    ŷ_uf = σ(MLP([h_u || h_f]))
    Inference: ŷ_uf^inference = σ(z + 0.1)
    """
    
    def __init__(self, hidden_channels, dropout=0.3, inference_bias=0.1):
        super().__init__()
        self.lin1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.inference_bias = inference_bias
    
    def forward(self, z_dict, edge_label_index, training=True):
        """
        Predict user-food interaction scores
        
        Args:
            z_dict: Encoded node embeddings
            edge_label_index: User-food edge indices
            training: Training mode flag
            
        Returns:
            torch.Tensor: Prediction scores (0-1)
        """
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['food'][col]], dim=-1)
        
        # MLP
        z = self.lin1(z)
        z = self.bn1(z)
        z = F.relu(z)
        z = self.dropout(z)
        z = self.lin2(z)
        
        # Inference bias (건강한 선택 유도)
        if training:
            return torch.sigmoid(z.view(-1))
        else:
            return torch.sigmoid((z + self.inference_bias).view(-1))


class NutriGraphNet(torch.nn.Module):
    """
    Complete NutriGraphNet Model
    
    논문의 전체 아키텍처:
    - Heterogeneous Graph Attention Encoder
    - Edge Decoder with Inference Bias
    - Health Attention Mechanism
    """
    
    def __init__(self, hidden_channels, out_channels, metadata, dropout=0.3, device=None):
        super().__init__()
        self.encoder = NutriGraphNetEncoder(hidden_channels, out_channels, metadata, dropout)
        self.decoder = NutriGraphNetDecoder(out_channels, dropout)
        self.device = device
    
    def forward(self, x_dict, edge_index_dict, edge_label_index, 
                health_edge_index=None, health_scores=None, training=True):
        """
        Full forward pass
        
        Args:
            x_dict: Node features
            edge_index_dict: All edge indices
            edge_label_index: Target user-food edges for prediction
            health_edge_index: Health relationship edges
            health_scores: Personalized health scores
            training: Training mode
            
        Returns:
            tuple: (predictions, user_health_preferences)
        """
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


class DualObjectiveLoss(nn.Module):
    """
    논문의 Dual-Objective Loss Function
    
    L = L_BCE(ŷ, y) + λ_h × L_Health
    L_Health = -mean(ŷ ⊙ health_scores)
    """
    
    def __init__(self, lambda_health=0.1, pos_weight=None):
        super().__init__()
        self.lambda_health = lambda_health
        self.pos_weight = pos_weight
    
    def forward(self, predictions, targets, health_scores, user_health_preferences=None):
        """
        Calculate dual-objective loss
        
        Args:
            predictions: Model predictions (0-1)
            targets: True labels (0 or 1)
            health_scores: Health scores (0-1)
            user_health_preferences: User health preference weights (optional)
            
        Returns:
            torch.Tensor: Total loss
        """
        device = predictions.device
        
        # 1. Preference Loss (BCE)
        if self.pos_weight is not None:
            weight = targets * (self.pos_weight - 1) + 1
            bce_loss = F.binary_cross_entropy(predictions, targets, weight=weight)
        else:
            bce_loss = F.binary_cross_entropy(predictions, targets)
        
        # 2. Health-Aware Regularization
        # 건강한 음식이 높은 예측값을 가지도록 유도
        health_loss = -torch.mean(predictions * health_scores)
        
        # 3. User-weighted Health Loss (optional)
        if user_health_preferences is not None:
            user_weights = torch.sigmoid(user_health_preferences.squeeze())
            weighted_health_loss = -torch.mean(predictions * health_scores * user_weights)
            health_loss = 0.5 * health_loss + 0.5 * weighted_health_loss
        
        # Total Loss
        total_loss = bce_loss + self.lambda_health * health_loss
        
        return total_loss
    
    def get_components(self, predictions, targets, health_scores):
        """Loss 구성요소 반환 (디버깅용)"""
        bce_loss = F.binary_cross_entropy(predictions, targets)
        health_loss = -torch.mean(predictions * health_scores)
        
        return {
            'bce_loss': bce_loss.item(),
            'health_loss': health_loss.item(),
            'total_loss': (bce_loss + self.lambda_health * health_loss).item()
        }


# 기존 LightGCN 호환성 유지 (비교 실험용)
class LightGCNEncoder(torch.nn.Module):
    """LightGCN baseline for comparison"""
    
    def __init__(self, hidden_channels, out_channels, metadata, n_layers=3, dropout=0.5, 
                 num_users=None, num_foods=None):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        
        self.embedding_dict = nn.ModuleDict({
            'user': nn.Embedding(num_users, hidden_channels),
            'food': nn.Embedding(num_foods, hidden_channels)
        })
        
        for node_type in ['user', 'food']:
            nn.init.normal_(self.embedding_dict[node_type].weight, std=0.1)
    
    def forward(self, x_dict, edge_index_dict, edge_index, batch_health_scores=None):
        device = self.embedding_dict['user'].weight.device
        
        user_idx, food_idx = edge_index_dict[('user', 'eats', 'food')]
        user_idx = user_idx.to(device)
        food_idx = food_idx.to(device)
        
        user_emb = self.embedding_dict['user'].weight
        food_emb = self.embedding_dict['food'].weight
        
        user_emb_list = [user_emb]
        food_emb_list = [food_emb]
        
        for i in range(self.n_layers):
            new_food_emb = self.aggregate(food_emb, user_emb, user_idx, food_idx)
            new_user_emb = self.aggregate(user_emb, food_emb, food_idx, user_idx)
            
            user_emb = new_user_emb
            food_emb = new_food_emb
            
            user_emb_list.append(user_emb)
            food_emb_list.append(food_emb)
        
        user_emb = torch.stack(user_emb_list, dim=0).mean(dim=0)
        food_emb = torch.stack(food_emb_list, dim=0).mean(dim=0)
        
        z_dict = {'user': user_emb, 'food': food_emb}
        
        return z_dict
    
    def aggregate(self, target_emb, source_emb, source_idx, target_idx):
        device = target_emb.device
        source_idx = source_idx.to(device)
        target_idx = target_idx.to(device)
        
        target_degrees = torch.zeros(target_emb.size(0), device=device)
        source_degrees = torch.zeros(source_emb.size(0), device=device)
        
        target_degrees.scatter_add_(0, target_idx, torch.ones_like(target_idx, dtype=torch.float))
        source_degrees.scatter_add_(0, source_idx, torch.ones_like(source_idx, dtype=torch.float))
        
        target_degrees = torch.pow(target_degrees + 1e-8, -0.5)
        source_degrees = torch.pow(source_degrees + 1e-8, -0.5)
        
        messages = source_emb[source_idx] * source_degrees[source_idx].unsqueeze(1)
        
        new_target_emb = torch.zeros_like(target_emb)
        new_target_emb.scatter_add_(0, target_idx.unsqueeze(1).expand(-1, source_emb.size(1)), messages)
        
        new_target_emb = new_target_emb * target_degrees.unsqueeze(1)
        return new_target_emb


class LightGCN(torch.nn.Module):
    """LightGCN baseline model"""
    
    def __init__(self, hidden_channels, out_channels, metadata, n_layers=3, dropout=0.5, 
                 device=None, num_users=None, num_foods=None):
        super().__init__()
        self.encoder = LightGCNEncoder(hidden_channels, out_channels, metadata, n_layers, 
                                       dropout, num_users, num_foods)
        self.decoder = NutriGraphNetDecoder(out_channels, dropout)
        self.device = device
    
    def forward(self, x_dict, edge_index_dict, edge_label_index, health_scores=None, training=True):
        edge_label_index = edge_label_index.long()
        edge_index_dict = {k: v.long() for k, v in edge_index_dict.items()}
        
        z_dict = self.encoder(x_dict, edge_index_dict, edge_label_index, health_scores)
        predictions = self.decoder(z_dict, edge_label_index, training=training)
        
        return predictions, None
