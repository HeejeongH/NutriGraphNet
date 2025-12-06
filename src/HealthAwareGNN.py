import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv

class GATEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.health_preference_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Tanh()  # -1 ~ 1 사이의 값을 생성 (-1: 비건강 선호, 1: 건강 선호)
        )

        self.skip = nn.ModuleDict({
            node_type: nn.Linear(hidden_channels, out_channels)
            for node_type in metadata[0]
        })

        conv = HeteroConv({
            ('user', 'healthness', 'food'): GATConv(
                in_channels=(-1, -1),
                out_channels=hidden_channels,
                heads=1,
                dropout=dropout,
                add_self_loops=False,
                edge_dim=1
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
        self.convs.append(conv)
        self.bns.append(nn.ModuleDict({
            node_type: nn.BatchNorm1d(hidden_channels)
            for node_type in metadata[0]
        }))

        conv = HeteroConv({
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
        self.convs.append(conv)
        self.bns.append(nn.ModuleDict({
            node_type: nn.BatchNorm1d(out_channels)
            for node_type in metadata[0]
        }))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict, edge_index, batch_health_scores, user_health_history=None):
        original_x = {k: v.clone() for k, v in x_dict.items()}

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.elu(bn[key](x)) for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
            
            if i == len(self.convs) - 1 and batch_health_scores is not None:
                user_indices = edge_index[0]
                food_indices = edge_index[1]
                
                user_health_preferences = self.health_preference_layer(x_dict['user'])  # -1 ~ 1 사이 값
                user_specific_weights = user_health_preferences[user_indices]
                
                adjusted_health_scores = user_specific_weights * batch_health_scores.unsqueeze(-1)
                
                food_dim = x_dict['food'].size(1)
                health_weighted_values = adjusted_health_scores.expand(-1, food_dim)
                
                food_updates = torch.zeros_like(x_dict['food'])
                food_updates.scatter_add_(0, food_indices.unsqueeze(-1).expand(-1, food_dim), health_weighted_values)
                
                x_dict['food'] = x_dict['food'] + 0.1 * food_updates
                
        return x_dict, user_health_preferences
   
class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, dropout=0.5, inference_bias=0.1):
        super().__init__()
        self.lin1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.inference_bias = inference_bias

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['food'][col]], dim=-1)
        
        z = self.lin1(z)
        z = self.bn1(z)
        z = F.relu(z)
        z = self.dropout(z)
        z = self.lin2(z)

        if self.training:
            return torch.sigmoid(z.view(-1))
        else:
            return torch.sigmoid((z + self.inference_bias).view(-1))

class HeteroGAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, dropout=0.5, device=None):
        super().__init__()
        self.encoder = GATEncoder(hidden_channels, out_channels, metadata, dropout)
        self.decoder = EdgeDecoder(out_channels, dropout)
        self.device = device
        
    def forward(self, x_dict, edge_index_dict, edge_label_index, health_scores=None, user_health_history=None):
        edge_label_index = edge_label_index.long()
        edge_index_dict = {k: v.long() for k, v in edge_index_dict.items()}
    
        z_dict, user_health_preferences = self.encoder(x_dict, edge_index_dict, edge_label_index, health_scores, user_health_history)
        return self.decoder(z_dict, edge_label_index), user_health_preferences
        
class HealthAwareCriterion(nn.Module):
    def __init__(self, lambda_health=0.001, pos_weight=1.5):
        super().__init__()
        self.lambda_health = lambda_health
        self.pos_weight = pos_weight
        
    def forward(self, pred, true, health_scores, user_health_preferences=None):
        eps = 1e-7
        pos_mask = (true > 0.5)
        neg_mask = ~pos_mask
        
        pos_loss = -torch.mean(torch.log(pred[pos_mask] + eps) * self.pos_weight)
        neg_loss = -torch.mean(torch.log(1 - pred[neg_mask] + eps))
        
        preference_loss = pos_loss + neg_loss if pos_mask.any() and neg_mask.any() else torch.tensor(0.0).to(pred.device)
        
        health_loss = -torch.mean(pred * health_scores)
        
        return preference_loss + self.lambda_health * health_loss

class LightGCNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, n_layers=3, dropout=0.5, num_users=None, num_foods=None):
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
    def __init__(self, hidden_channels, out_channels, metadata, n_layers=3, dropout=0.5, device=None, num_users=None, num_foods=None):
        super().__init__()
        self.encoder = LightGCNEncoder(hidden_channels, out_channels, metadata, n_layers, dropout, num_users, num_foods)
        self.decoder = EdgeDecoder(out_channels, dropout)
        self.device = device

    def forward(self, x_dict, edge_index_dict, edge_label_index, health_scores=None):
        edge_label_index = edge_label_index.long()
        edge_index_dict = {k: v.long() for k, v in edge_index_dict.items()}
    
        z_dict = self.encoder(x_dict, edge_index_dict, edge_label_index, health_scores)
        return self.decoder(z_dict, edge_label_index)