"""
완전히 수정된 그래프 데이터 생성 스크립트
모든 edge weight와 health score를 올바르게 정규화합니다.
"""

import pandas as pd
import numpy as np
import warnings
import pyreadstat
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import sys
import os

# Health score calculator import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from health_score_calculator import PersonalizedHealthScoreCalculator


def normalize_weights(weights, method='log1p'):
    """
    Edge weight 정규화
    
    Args:
        weights: numpy array or torch tensor
        method: 'log1p', 'minmax', 'clip'
    
    Returns:
        정규화된 weights
    """
    if isinstance(weights, torch.Tensor):
        weights = weights.numpy()
    
    if method == 'log1p':
        # Log transformation: log(1 + x)
        normalized = np.log1p(weights)
        # 0-1 범위로 추가 정규화
        if normalized.max() > 0:
            normalized = normalized / normalized.max()
    elif method == 'minmax':
        # Min-max normalization
        if weights.max() > weights.min():
            normalized = (weights - weights.min()) / (weights.max() - weights.min())
        else:
            normalized = np.ones_like(weights) * 0.5
    elif method == 'clip':
        # Clipping outliers
        normalized = np.clip(weights, 0, 10)
        if normalized.max() > 0:
            normalized = normalized / 10.0
    else:
        normalized = weights
    
    return normalized


def load_data(file_path):
    """데이터 로드 및 전처리"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"🔧 Graph Data Builder - 완전 수정 버전")
    print(f"{'='*70}")
    print(f"Device: {device}")
    
    warnings.filterwarnings('ignore')
    
    print(f"\n📂 Loading raw data from: {file_path}")
    
    # 데이터 불러오기
    diet_raw_data, diet_meta = pyreadstat.read_sav(file_path)
    
    print(f"✅ Raw data loaded: {len(diet_raw_data):,} rows")
    
    # 원하는 컬럼만 추출
    imp_column = ['ID','sex','age','incm','edu','occp','N_DCODE', 'N_DNAME', 'N_MEAL', 
                  'N_FCODE', 'N_FNAME', 'N_TD_VOL', 'N_TD_WT', 'N_CD_VOL', 'N_CD_WT', 
                  'N_KINDG1', 'N_FM_WT']
    nutri_column = ['NF_EN','NF_CHO','NF_PROT','NF_FAT','NF_SFA','NF_CHOL',
                   'NF_TDF','NF_SUGAR','NF_NA','NF_CA','NF_PHOS','NF_K']
    
    diet_data = diet_raw_data[imp_column + nutri_column].copy()
    
    # 부피, 중량 합치기
    diet_data['N_TD'] = np.where(diet_data['N_TD_VOL'].notna(), diet_data['N_TD_VOL'], diet_data['N_TD_WT'])
    diet_data['N_CD'] = np.where(diet_data['N_CD_VOL'].notna(), diet_data['N_CD_VOL'], diet_data['N_CD_WT'])
    diet_data = diet_data.drop(['N_TD_VOL','N_TD_WT','N_CD_VOL','N_CD_WT'], axis=1)
    diet_data['N_CD'] = np.where(diet_data['N_CD'].notna(), diet_data['N_CD'], diet_data['N_TD'])
    diet_data['N_TD'] = np.where(diet_data['N_TD'].notna(), diet_data['N_TD'], diet_data['N_CD'])
    
    # 음식 코드 통일
    diet_data = diet_data.rename(columns={'N_DCODE':'O_DCODE'})
    name_to_code = {name: f"{i:02d}" for i, name in enumerate(diet_data['N_DNAME'].unique())}
    diet_data['N_DCODE'] = diet_data['O_DCODE'].astype(str) + diet_data['N_DNAME'].map(name_to_code)
    diet_data = diet_data.drop(columns=['O_DCODE']).drop_duplicates().reset_index(drop=True)
    diet_data = diet_data.fillna(0)
    diet_data['N_KINDG1'] = diet_data['N_KINDG1'].astype(int)
    
    print(f"✅ Data preprocessed: {len(diet_data):,} rows")
    
    # ========================================
    # 노드 데이터 준비
    # ========================================
    
    print(f"\n{'='*70}")
    print("📊 Building Node Features")
    print(f"{'='*70}")
    
    # User 노드
    user_data = diet_data[['ID', 'sex', 'age', 'incm', 'edu', 'occp']].drop_duplicates()
    print(f"  Users: {len(user_data):,}")
    
    # Food 노드
    food_data = diet_data[['N_DCODE','N_TD','N_CD'] + nutri_column].drop_duplicates().reset_index(drop=True)
    
    # 🔧 중요: 영양소를 섭취량 기준으로 계산
    for nutri in nutri_column:
        food_data[nutri] = food_data['N_CD'] * food_data[nutri] / food_data['N_TD']
    
    food_data = food_data.drop(['N_TD','N_CD'], axis=1)
    food_data = food_data.groupby('N_DCODE').sum().reset_index()
    print(f"  Foods: {len(food_data):,}")
    
    # Ingredient 노드
    ingre_data = diet_data[['N_FCODE', 'N_FNAME', 'N_KINDG1']].drop_duplicates().set_index('N_FCODE')
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(ingre_data['N_FNAME'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df.index = ingre_data.index
    ingre_data = ingre_data.join(tfidf_df).drop('N_FNAME', axis=1).reset_index()
    print(f"  Ingredients: {len(ingre_data):,}")
    
    # Time 노드
    meal_data = diet_data[['N_MEAL']].drop_duplicates()
    print(f"  Time nodes: {len(meal_data):,}")
    
    # ========================================
    # 엣지 데이터 준비
    # ========================================
    
    print(f"\n{'='*70}")
    print("🔗 Building Edges")
    print(f"{'='*70}")
    
    # 1. User-eats-Food
    print(f"\n1️⃣ User-eats-Food edges:")
    intake_data = diet_data[['ID', 'N_DCODE','N_TD','N_CD']]
    
    # 🔧 수정: Edge weight를 섭취 빈도로 계산
    intake_data['intake_count'] = 1  # 각 섭취 기록을 카운트
    intake_grouped = intake_data.groupby(['ID','N_DCODE']).agg({
        'intake_count': 'sum',  # 섭취 횟수
        'N_TD': 'sum',  # 총 섭취량
        'N_CD': 'mean'  # 평균 1회 제공량
    }).reset_index()
    
    # Edge weight = 섭취 횟수 (정규화 예정)
    intake_grouped['edge_weight'] = intake_grouped['intake_count'].astype(float)
    
    print(f"   Raw edges: {len(intake_grouped):,}")
    print(f"   Weight range (before): [{intake_grouped['edge_weight'].min():.2f}, {intake_grouped['edge_weight'].max():.2f}]")
    
    # 🔧 정규화 적용
    intake_grouped['edge_weight'] = normalize_weights(intake_grouped['edge_weight'].values, method='log1p')
    
    print(f"   Weight range (after): [{intake_grouped['edge_weight'].min():.4f}, {intake_grouped['edge_weight'].max():.4f}]")
    print(f"   Weight mean: {intake_grouped['edge_weight'].mean():.4f}")
    
    # 2. Food-similar-Food (함께 먹은 음식)
    print(f"\n2️⃣ Food-similar-Food edges:")
    def create_food_pairs(meal_data):
        pairs = []
        for _, group in meal_data.groupby(['ID', 'N_MEAL']):
            foods = group['N_DCODE'].tolist()
            pairs.extend([(food1, food2) for i, food1 in enumerate(foods) for food2 in foods[i+1:]])
        return pd.DataFrame(pairs, columns=['food1', 'food2']).drop_duplicates()
    
    meal_data_full = diet_data[['ID', 'N_DCODE', 'N_MEAL']].drop_duplicates()
    food_pairs = create_food_pairs(meal_data_full)
    print(f"   Pairs: {len(food_pairs):,}")
    
    # 3. Food-contains-Ingredient
    print(f"\n3️⃣ Food-contains-Ingredient edges:")
    food_ing_data = diet_data[['N_DCODE', 'N_FCODE', 'N_FM_WT']].drop_duplicates()
    food_ing_data = food_ing_data.groupby(['N_DCODE','N_FCODE']).mean().reset_index()
    
    print(f"   Raw weight range: [{food_ing_data['N_FM_WT'].min():.2f}, {food_ing_data['N_FM_WT'].max():.2f}]")
    
    # 🔧 정규화
    food_ing_data['edge_weight'] = normalize_weights(food_ing_data['N_FM_WT'].values, method='log1p')
    
    print(f"   Normalized range: [{food_ing_data['edge_weight'].min():.4f}, {food_ing_data['edge_weight'].max():.4f}]")
    
    # 4. Food-eaten_at-Time
    print(f"\n4️⃣ Food-eaten_at-Time edges:")
    food_time_data = diet_data.groupby(['N_DCODE', 'N_MEAL']).size().reset_index(name='count')
    # 음식별로 시간대 비율 계산 (이미 0-1 범위)
    food_time_data['edge_weight'] = food_time_data.groupby('N_DCODE')['count'].transform(lambda x: x / x.sum())
    print(f"   Edges: {len(food_time_data):,}")
    print(f"   Weight range: [{food_time_data['edge_weight'].min():.4f}, {food_time_data['edge_weight'].max():.4f}]")
    
    # ========================================
    # HeteroData 구성
    # ========================================
    
    print(f"\n{'='*70}")
    print("🏗️ Building HeteroData")
    print(f"{'='*70}")
    
    def load_node_csv(data, index_col, feature=None):
        df = data.set_index(index_col)
        mapping = {index: i for i, index in enumerate(df.index.unique())}
        x = None
        if feature is not None:
            x = torch.from_numpy(df.values).to(torch.float)
        return x, mapping
    
    def load_edge_csv(df, src_index_col, src_mapping, dst_index_col, dst_mapping, weight_col=None):
        src = [src_mapping[index] for index in df[src_index_col]]
        dst = [dst_mapping[index] for index in df[dst_index_col]]
        edge_index = torch.tensor([src, dst])
        
        edge_attr = None
        if weight_col is not None and weight_col in df.columns:
            edge_attr = torch.tensor(df[weight_col].values, dtype=torch.float32)
        
        return edge_index, edge_attr
    
    # 노드 로드
    user_x, user_mapping = load_node_csv(user_data, 'ID', feature=not None)
    food_x, food_mapping = load_node_csv(food_data, 'N_DCODE', feature=not None)
    ingre_x, ingre_mapping = load_node_csv(ingre_data, 'N_FCODE', feature=not None)
    meal_x, meal_mapping = load_node_csv(meal_data, 'N_MEAL', feature=None)
    
    print(f"  Node mappings created")
    
    # 엣지 로드
    uf_edge_index, uf_edge_attr = load_edge_csv(
        intake_grouped, 'ID', user_mapping, 'N_DCODE', food_mapping, 
        weight_col='edge_weight'
    )
    ff_edge_index, ff_edge_attr = load_edge_csv(
        food_pairs, 'food1', food_mapping, 'food2', food_mapping
    )
    fi_edge_index, fi_edge_attr = load_edge_csv(
        food_ing_data, 'N_DCODE', food_mapping, 'N_FCODE', ingre_mapping, 
        weight_col='edge_weight'
    )
    ft_edge_index, ft_edge_attr = load_edge_csv(
        food_time_data, 'N_DCODE', food_mapping, 'N_MEAL', meal_mapping, 
        weight_col='edge_weight'
    )
    
    print(f"  Edges loaded")
    
    # HeteroData 생성
    data = HeteroData()
    
    # 노드 설정
    data["user"].node_id = torch.arange(len(user_mapping))
    data["food"].node_id = torch.arange(len(food_mapping))
    data["ingredient"].node_id = torch.arange(len(ingre_mapping))
    data["time"].node_id = torch.arange(len(meal_mapping))
    
    data['user'].x = user_x
    data['food'].x = food_x
    data['ingredient'].x = ingre_x
    
    # 엣지 설정
    data["user", "eats", "food"].edge_index = uf_edge_index
    data["user", "eats", "food"].edge_attr = uf_edge_attr
    
    data["food", "similar", "food"].edge_index = ff_edge_index
    
    data["food", "contains", "ingredient"].edge_index = fi_edge_index
    data["food", "contains", "ingredient"].edge_attr = fi_edge_attr
    
    data["food", "eaten_at", "time"].edge_index = ft_edge_index
    data["food", "eaten_at", "time"].edge_attr = ft_edge_attr
    
    print(f"  HeteroData created")
    
    # 역방향 엣지 추가
    print(f"\n  Adding reverse edges...")
    data = T.ToUndirected()(data)
    
    # edge_label 제거 (필요 없음)
    if hasattr(data['food', 'rev_eats', 'user'], 'edge_label'):
        del data['food', 'rev_eats', 'user'].edge_label
    
    # ========================================
    # Health Score 계산
    # ========================================
    
    print(f"\n{'='*70}")
    print("💚 Computing Health Scores")
    print(f"{'='*70}")
    
    calculator = PersonalizedHealthScoreCalculator()
    
    # User-food 엣지에 대한 건강 점수 계산
    user_indices = uf_edge_index[0].numpy()
    food_indices = uf_edge_index[1].numpy()
    n_edges = len(user_indices)
    
    health_scores = np.zeros(n_edges)
    
    print(f"Computing health scores for {n_edges:,} edges...")
    
    # 배치 처리
    batch_size = 10000
    for start_idx in range(0, n_edges, batch_size):
        end_idx = min(start_idx + batch_size, n_edges)
        
        for i in range(start_idx, end_idx):
            u_idx = user_indices[i]
            f_idx = food_indices[i]
            
            user_feat = user_x[u_idx].numpy()
            food_feat = food_x[f_idx].numpy()
            
            # 사용자 특성 추출
            age = max(user_feat[1], 1) if len(user_feat) > 1 else 30  # sex, age, ...
            gender = 'male' if user_feat[0] > 0.5 else 'female' if len(user_feat) > 0 else 'male'
            height = 170  # 기본값 (데이터에 없음)
            weight = 70   # 기본값
            
            # EER 계산
            eer = calculator.calculate_eer(age, gender, height, weight, 'low_active')
            standards = calculator.calculate_personalized_standards(eer)
            
            # 음식 영양소 매핑 (NF_EN, NF_CHO, NF_PROT, NF_FAT, NF_SFA, NF_CHOL, NF_TDF, NF_SUGAR, NF_NA, ...)
            food_nutrients = {
                'energy': food_feat[0] if len(food_feat) > 0 else 0,
                'protein': food_feat[2] if len(food_feat) > 2 else 0,
                'fat': food_feat[3] if len(food_feat) > 3 else 0,
                'saturated_fat': food_feat[4] if len(food_feat) > 4 else 0,
                'cholesterol': food_feat[5] if len(food_feat) > 5 else 0,
                'fiber': food_feat[6] if len(food_feat) > 6 else 0,
                'sugar': food_feat[7] if len(food_feat) > 7 else 0,
                'sodium': food_feat[8] if len(food_feat) > 8 else 0,
            }
            
            score = calculator.calculate_food_health_score(food_nutrients, standards)
            health_scores[i] = score
        
        if (start_idx // batch_size) % 5 == 0:
            progress = 100 * end_idx / n_edges
            print(f"  Progress: {end_idx:,}/{n_edges:,} ({progress:.1f}%)")
    
    # Health 엣지 추가
    health_scores_tensor = torch.tensor(health_scores, dtype=torch.float32)
    
    data["user", "healthness", "food"].edge_index = uf_edge_index
    data["user", "healthness", "food"].edge_attr = health_scores_tensor
    
    # 역방향도 추가
    data["food", "rev_healthness", "user"].edge_index = torch.stack([uf_edge_index[1], uf_edge_index[0]])
    data["food", "rev_healthness", "user"].edge_attr = health_scores_tensor
    
    print(f"\n✅ Health scores computed!")
    print(f"   Range: [{health_scores.min():.4f}, {health_scores.max():.4f}]")
    print(f"   Mean: {health_scores.mean():.4f}")
    print(f"   Median: {np.median(health_scores):.4f}")
    
    # ========================================
    # 최종 요약
    # ========================================
    
    print(f"\n{'='*70}")
    print("📊 Final Data Summary")
    print(f"{'='*70}")
    
    print(f"\n✅ Nodes:")
    print(f"   Users: {len(user_mapping):,}")
    print(f"   Foods: {len(food_mapping):,}")
    print(f"   Ingredients: {len(ingre_mapping):,}")
    print(f"   Time: {len(meal_mapping):,}")
    
    print(f"\n✅ Edges:")
    for edge_type in data.edge_types:
        edge_count = data[edge_type].edge_index.shape[1]
        has_attr = hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None
        
        if has_attr:
            attr = data[edge_type].edge_attr
            print(f"   {edge_type}: {edge_count:,} edges, weights [{attr.min():.4f}, {attr.max():.4f}]")
        else:
            print(f"   {edge_type}: {edge_count:,} edges")
    
    # 매핑 정보 저장
    data.user_id_mapping = user_mapping
    data.food_id_mapping = food_mapping
    data.ingredient_id_mapping = ingre_mapping
    data.time_id_mapping = meal_mapping
    
    return data


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Build graph data from raw SAV file')
    parser.add_argument('--input', type=str, 
                       default='../data/raw/HN22_ALL.sav',
                       help='Input SAV file path')
    parser.add_argument('--output', type=str, 
                       default='../data/processed_data/processed_data_GNN_fixed.pkl',
                       help='Output pickle file path')
    
    args = parser.parse_args()
    
    # 데이터 생성
    data = load_data(args.input)
    
    # CPU로 이동 (저장용)
    data = data.to('cpu')
    
    # 저장
    print(f"\n💾 Saving data to: {args.output}")
    
    # 출력 디렉토리 생성
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✅ Data saved successfully!")
    print(f"\nFile size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")
    print(f"\n{'='*70}")
    print("🎉 All done! You can now use this data for training.")
    print(f"{'='*70}\n")
