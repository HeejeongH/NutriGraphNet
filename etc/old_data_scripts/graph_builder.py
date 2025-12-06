import pandas as pd
import numpy as np
import warnings
import pyreadstat

import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")

    warnings.filterwarnings('ignore')

    # 데이터 불러오기
    diet_raw_data, diet_meta = pyreadstat.read_sav(file_path)
    diet_meta_df = pd.DataFrame([diet_meta.column_names, diet_meta.column_labels]).transpose()

    diet_raw_data = diet_raw_data

    # 원하는 데이터만 추출
    imp_column = ['ID','sex','age','incm','edu','occp','N_DCODE', 'N_DNAME', 'N_MEAL', 'N_FCODE', 'N_FNAME', 'N_TD_VOL', 'N_TD_WT', 'N_CD_VOL', 'N_CD_WT', 'N_KINDG1', 'N_FM_WT']
    nutri_column = ['NF_EN','NF_CHO','NF_PROT','NF_FAT','NF_SFA','NF_CHOL','NF_TDF','NF_SUGAR','NF_NA','NF_CA','NF_PHOS','NF_K']

    diet_data = diet_raw_data[imp_column+nutri_column].copy()

    # 부피, 중량 합치기
    diet_data['N_TD'] = np.where(diet_data['N_TD_VOL'].notna(), diet_data['N_TD_VOL'], diet_data['N_TD_WT'])
    diet_data['N_CD'] = np.where(diet_data['N_CD_VOL'].notna(), diet_data['N_CD_VOL'], diet_data['N_CD_WT'])
    diet_data = diet_data.drop(['N_TD_VOL','N_TD_WT'], axis=1)  
    diet_data = diet_data.drop(['N_CD_VOL','N_CD_WT'], axis=1)
    diet_data['N_CD'] = np.where(diet_data['N_CD'].notna(), diet_data['N_CD'], diet_data['N_TD'])
    diet_data['N_TD'] = np.where(diet_data['N_TD'].notna(), diet_data['N_TD'], diet_data['N_CD'])

    # 음식 코드 통일
    diet_data = diet_data.rename(columns={'N_DCODE':'O_DCODE'})
    name_to_code = {name: f"{i:02d}" for i, name in enumerate(diet_data['N_DNAME'].unique())}
    diet_data['N_DCODE'] = diet_data['O_DCODE'].astype(str) + diet_data['N_DNAME'].map(name_to_code)
    diet_data = diet_data.drop(columns=['O_DCODE']).drop_duplicates().reset_index(drop=True)
    diet_data = diet_data.fillna(0)

    # ID 숫자로 매핑해주기
    diet_data['N_KINDG1'] = diet_data['N_KINDG1'].astype(int)

    # 각 데이터 정의
    user_data = diet_data[['ID', 'sex', 'age', 'incm', 'edu', 'occp']].drop_duplicates()

    food_data = diet_data[['N_DCODE','N_TD','N_CD']+nutri_column].drop_duplicates().reset_index(drop=True)
    for nutri in nutri_column:
        food_data[nutri] = food_data['N_CD'] * food_data[nutri] / food_data['N_TD']
    food_data = food_data.drop(['N_TD','N_CD'], axis=1)
    food_data = food_data.groupby('N_DCODE').sum().reset_index()

    ingre_data = diet_data[['N_FCODE', 'N_FNAME', 'N_KINDG1']].drop_duplicates().set_index('N_FCODE')
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(ingre_data['N_FNAME'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df.index = ingre_data.index
    ingre_data = ingre_data.join(tfidf_df).drop('N_FNAME', axis=1).reset_index()

    meal_data = diet_data[['N_MEAL']].drop_duplicates()

    # user - food
    intake_data = diet_data[['ID', 'N_DCODE','N_TD','N_CD']]
    intake_data['edge_weight'] = intake_data['N_TD']/intake_data['N_CD']
    intake_data = intake_data.drop(['N_TD','N_CD'], axis=1).drop_duplicates().reset_index(drop=True)
    intake_data = intake_data.groupby(['ID','N_DCODE']).mean().reset_index()

    # food - food
    def create_food_pairs(meal_data):
        pairs = []
        for _, group in meal_data.groupby(['ID', 'N_MEAL']):
            foods = group['N_DCODE'].tolist()
            pairs.extend([(food1, food2) for i, food1 in enumerate(foods) for food2 in foods[i+1:]])
        return pd.DataFrame(pairs, columns=['food1', 'food2']).drop_duplicates()

    meal_data = diet_data[['ID', 'N_DCODE', 'N_MEAL']].drop_duplicates().reset_index(drop=True)
    food_pairs = create_food_pairs(meal_data)

    # food - ingredient
    food_ing_data = diet_data[['N_DCODE', 'N_FCODE', 'N_FM_WT']].drop_duplicates().reset_index(drop=True)
    food_ing_data = food_ing_data.groupby(['N_DCODE','N_FCODE']).mean().reset_index()
    food_ing_data = food_ing_data.rename(columns={'N_FM_WT':'edge_weight'})

    # food - time
    food_time_data = diet_data.groupby(['N_DCODE', 'N_MEAL']).size().reset_index(name='count')
    food_time_data['edge_weight'] = food_time_data.groupby('N_DCODE')['count'].transform(lambda x: x / x.sum())
    food_time_data = food_time_data.drop('count', axis=1)

    def load_node_csv(data, index_col, feature=None, **kwargs):
        df = data.set_index(index_col)
        mapping = {index: i for i, index in enumerate(df.index.unique())}

        x = None
        if feature is not None:
            x = torch.from_numpy(df.values).to(torch.float)
        return x, mapping

    def load_edge_csv(df, src_index_col, src_mapping, dst_index_col, dst_mapping, weight_col=None, **kwargs):
        src = [src_mapping[index] for index in df[src_index_col]]
        dst = [dst_mapping[index] for index in df[dst_index_col]]
        edge_index = torch.tensor([src, dst])

        edge_attr = None
        if weight_col is not None:
            edge_attr = df[weight_col]

        return edge_index, edge_attr

    user_x, user_mapping = load_node_csv(user_data, 'ID', feature = not None)
    food_x, food_mapping = load_node_csv(food_data, 'N_DCODE', feature = not None)
    ingre_x, ingre_mapping = load_node_csv(ingre_data, 'N_FCODE', feature = not None)
    meal_x, meal_mapping = load_node_csv(meal_data, 'N_MEAL', feature = None)

    uf_edge_index, uf_edge_attr = load_edge_csv(intake_data, 'ID', user_mapping, 'N_DCODE', food_mapping, weight_col='edge_weight')
    ff_edge_index, ff_edge_attr = load_edge_csv(food_pairs, 'food1', food_mapping, 'food2', food_mapping)
    fi_edge_index, fi_edge_attr = load_edge_csv(food_ing_data, 'N_DCODE', food_mapping, 'N_FCODE', ingre_mapping, weight_col='edge_weight')
    ft_edge_index, ft_edge_attr = load_edge_csv(food_time_data, 'N_DCODE', food_mapping, 'N_MEAL', meal_mapping, weight_col='edge_weight')

    data = HeteroData()

    # 노드 넣기
    data["user"].node_id = torch.tensor(list(user_mapping.values()), dtype=torch.long).to(device)
    data["food"].node_id = torch.tensor(list(food_mapping.values()), dtype=torch.long).to(device)
    data["ingredient"].node_id = torch.tensor(list(ingre_mapping.values()), dtype=torch.long).to(device)
    data["time"].node_id = torch.tensor(list(meal_mapping.values()), dtype=torch.long).to(device)

    # 노드 특성 설정
    data['user'].x = user_x.to(device)
    data['food'].x = food_x.to(device)
    data['ingredient'].x = ingre_x.to(device)

    data["user", "eats", "food"].edge_index = uf_edge_index.to(device)
    data["food", "pairs", "food"].edge_index = ff_edge_index.to(device)
    data["food", "contains", "ingredient"].edge_index = fi_edge_index.to(device)
    data["food", "eaten_at", "time"].edge_index = ft_edge_index.to(device)

    data = T.ToUndirected()(data)
    del data['food', 'rev_eats', 'user'].edge_label
    
    return data