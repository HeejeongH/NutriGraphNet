"""
간단한 샘플 데이터 생성 스크립트
테스트 및 데모용
"""

import numpy as np
import torch
import pickle
from pathlib import Path
import sys
sys.path.append('src')
from simple_hetero_data import SimpleHeteroData

def create_sample_data(
    num_users=1000,
    num_foods=1000,
    num_ingredients=100,
    num_interactions=10000,
    save_path='../data/processed_data/processed_data_GNN.pkl'
):
    """
    테스트용 샘플 데이터 생성
    
    Args:
        num_users: 사용자 수
        num_foods: 음식 수
        num_ingredients: 재료 수
        num_interactions: 사용자-음식 상호작용 수
        save_path: 저장 경로
    """
    
    print("📊 Creating sample data...")
    print(f"   Users: {num_users:,}")
    print(f"   Foods: {num_foods:,}")
    print(f"   Ingredients: {num_ingredients:,}")
    print(f"   Interactions: {num_interactions:,}")
    
    # 노드 특성 생성
    user_features = np.random.randn(num_users, 29).astype(np.float32)
    food_features = np.random.randn(num_foods, 17).astype(np.float32)
    ingredient_features = np.random.randn(num_ingredients, 101).astype(np.float32)
    time_features = np.eye(4, dtype=np.float32)
    
    # 사용자-음식 상호작용
    user_indices = np.random.randint(0, num_users, num_interactions)
    food_indices = np.random.randint(0, num_foods, num_interactions)
    eats_scores = np.random.beta(2, 2, num_interactions).astype(np.float32)
    
    # 건강 점수 (음식 특성 기반)
    health_scores = 0.3 + 0.6 * np.random.beta(5, 2, num_interactions).astype(np.float32)
    
    # 음식-재료 연결
    num_food_ing = min(5000, num_foods * 5)
    food_ing_food = np.random.randint(0, num_foods, num_food_ing)
    food_ing_ing = np.random.randint(0, num_ingredients, num_food_ing)
    
    # 음식-시간 연결
    num_food_time = min(3000, num_foods * 3)
    food_time_food = np.random.randint(0, num_foods, num_food_time)
    food_time_time = np.random.randint(0, 4, num_food_time)
    
    # HeteroData 생성
    data = SimpleHeteroData()
    
    # 노드 특성
    data.x_dict = {
        'user': torch.FloatTensor(user_features),
        'food': torch.FloatTensor(food_features),
        'ingredient': torch.FloatTensor(ingredient_features),
        'time': torch.FloatTensor(time_features)
    }
    
    # 엣지 인덱스
    eats_edge = torch.LongTensor(np.stack([user_indices, food_indices]))
    data.edge_index_dict[('user', 'eats', 'food')] = eats_edge
    data.edge_attr_dict[('user', 'eats', 'food')] = torch.FloatTensor(eats_scores)
    
    data.edge_index_dict[('food', 'rev_eats', 'user')] = torch.stack([
        eats_edge[1], eats_edge[0]
    ])
    
    # 건강 엣지
    health_edge = torch.LongTensor(np.stack([user_indices, food_indices]))
    data.edge_index_dict[('user', 'healthness', 'food')] = health_edge
    data.edge_attr_dict[('user', 'healthness', 'food')] = torch.FloatTensor(health_scores)
    
    data.edge_index_dict[('food', 'rev_healthness', 'user')] = torch.stack([
        health_edge[1], health_edge[0]
    ])
    
    # 음식-재료 엣지
    food_ing_edge = torch.LongTensor(np.stack([food_ing_food, food_ing_ing]))
    data.edge_index_dict[('food', 'contains', 'ingredient')] = food_ing_edge
    data.edge_index_dict[('ingredient', 'rev_contains', 'food')] = torch.stack([
        food_ing_edge[1], food_ing_edge[0]
    ])
    
    # 음식-시간 엣지
    food_time_edge = torch.LongTensor(np.stack([food_time_food, food_time_time]))
    data.edge_index_dict[('food', 'eaten_at', 'time')] = food_time_edge
    data.edge_index_dict[('time', 'rev_eaten_at', 'food')] = torch.stack([
        food_time_edge[1], food_time_edge[0]
    ])
    
    # 음식-음식 유사도
    num_similar = min(1000, num_foods)
    food_sim_1 = np.random.randint(0, num_foods, num_similar)
    food_sim_2 = np.random.randint(0, num_foods, num_similar)
    data.edge_index_dict[('food', 'similar', 'food')] = torch.LongTensor(
        np.stack([food_sim_1, food_sim_2])
    )
    
    # ID 매핑
    data.user_id_mapping = {i: i for i in range(num_users)}
    data.food_id_mapping = {i: i for i in range(num_foods)}
    
    # 저장
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n✅ Sample data created: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"\n🎉 Ready to train!")
    print(f"   Run: python train_v2.py --epochs 30")
    
    return data

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create sample data for testing')
    parser.add_argument('--num_users', type=int, default=1000, help='Number of users')
    parser.add_argument('--num_foods', type=int, default=1000, help='Number of foods')
    parser.add_argument('--num_ingredients', type=int, default=100, help='Number of ingredients')
    parser.add_argument('--num_interactions', type=int, default=10000, help='Number of interactions')
    parser.add_argument('--save_path', type=str, 
                       default='../data/processed_data/processed_data_GNN.pkl',
                       help='Save path')
    
    args = parser.parse_args()
    
    create_sample_data(
        num_users=args.num_users,
        num_foods=args.num_foods,
        num_ingredients=args.num_ingredients,
        num_interactions=args.num_interactions,
        save_path=args.save_path
    )
