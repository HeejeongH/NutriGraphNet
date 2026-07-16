"""
Food.com → NutriGraphNet HeteroData 전처리 스크립트

Input:
    data/foodcom/RAW_interactions.csv  (user_id, recipe_id, rating, ...)
    data/foodcom/RAW_recipes.csv       (id, nutrition, ingredients, ...)

Output:
    data/foodcom/processed_foodcom.pkl  (HeteroData, HFRS-DA 동일 포맷)

Graph Schema (NutriGraphNet 동일):
    Nodes: user, food, ingredient, time(dummy)
    Edges:
        user -[eats]-> food          (rating >= 4 → positive interaction)
        food -[rev_eats]-> user
        user -[healthness]-> food    (health_score edge_attr)
        food -[contains]-> ingredient
        ingredient -[rev_contains]-> food
        food -[similar]-> food       (same ingredient 공유, top-5 by jaccard)

Health Score:
    WHO 기준 7가지 영양소 기반 (HFRS-DA 동일 방식)
    nutrition: [calories, total_fat%, sugar%, sodium%, protein%, sat_fat%, carbs%]

필터링 (HFRS-DA 논문 기준):
    - rating >= 4 를 positive interaction으로 사용
    - 최소 5개 이상 interaction 있는 user 유지
    - 최소 5개 이상 interaction 있는 food 유지
    - 상위 1050개 ingredient만 사용 (HFRS-DA: 1050개)
"""

import os, sys, ast, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
warnings.filterwarnings('ignore')

import torch
from torch_geometric.data import HeteroData

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
INT_F  = BASE / 'foodcom' / 'RAW_interactions.csv'
REC_F  = BASE / 'foodcom' / 'RAW_recipes.csv'
OUT_F  = BASE / 'foodcom' / 'processed_foodcom.pkl'

# ── 하이퍼파라미터 ─────────────────────────────────────────────────────────────
MIN_USER_INTER = 5
MIN_FOOD_INTER = 5
MIN_RATING     = 4          # positive threshold
MAX_ING        = 1050       # HFRS-DA 기준
MAX_SIMILAR    = 5          # food-similar-food top-K (Jaccard)
FOOD_FEAT_DIM  = 17         # nutrition(7) + n_ingredients(1) + n_steps(1) + dummy(8)
USER_FEAT_DIM  = 29         # interaction stats + dummy padding

# ── WHO 건강 기준 (HFRS-DA Table 2) ──────────────────────────────────────────
# nutrition 컬럼: [calories, total_fat%, sugar%, sodium_PDV, protein%, sat_fat%, carbs%]
# PDV = % of daily value  →  healthy range 판단
WHO_RANGES = {
    # (idx, lower, upper, higher_is_better)
    'protein':   (4,  10,  15, True),   # 10~15% → good
    'carbs':     (6,  55,  75, True),   # 55~75% → good
    'sugar':     (2,   0,  10, False),  # <10%  → good (lower better)
    'sodium':    (3,   0,   5, False),  # <5g   → PDV 기준
    'fat':       (1,  15,  30, True),   # 15~30% → good
    'sat_fat':   (5,   0,  10, False),  # <10%  → good
}

def parse_list_col(x):
    """문자열로 된 list를 실제 Python list로 변환"""
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

def compute_health_score(nutrition_list):
    """
    WHO 기준 건강 점수 계산 (0~1 스케일).
    nutrition: [calories, total_fat%, sugar%, sodium_PDV, protein%, sat_fat%, carbs%]
    """
    if not nutrition_list or len(nutrition_list) < 7:
        return 0.5  # 정보 없으면 중간값

    nutr = [float(v) for v in nutrition_list]
    scores = []

    for name, (idx, lo, hi, higher_better) in WHO_RANGES.items():
        v = nutr[idx]
        if higher_better:
            # v가 [lo, hi] 안에 있을수록 좋음
            if v < lo:
                s = v / lo if lo > 0 else 0.5
            elif v <= hi:
                s = 1.0
            else:
                s = max(0.0, 1.0 - (v - hi) / hi) if hi > 0 else 0.0
        else:
            # v가 낮을수록 좋음 (< hi 기준)
            if v <= lo:
                s = 1.0
            elif v <= hi:
                s = 1.0 - (v - lo) / (hi - lo)
            else:
                s = max(0.0, 1.0 - (v - hi) / hi) if hi > 0 else 0.0
        scores.append(s)

    return float(np.mean(scores))


def main():
    print("=" * 60)
    print("  Food.com → NutriGraphNet HeteroData 전처리")
    print("=" * 60)

    # ── 1. 데이터 로드 ────────────────────────────────────────────────────────
    print("\n[1] Loading raw data...")
    interactions = pd.read_csv(INT_F)
    recipes      = pd.read_csv(REC_F)
    print(f"    interactions: {interactions.shape}")
    print(f"    recipes:      {recipes.shape}")

    # ── 2. Positive interaction 필터 ──────────────────────────────────────────
    print(f"\n[2] Filtering positive interactions (rating >= {MIN_RATING})...")
    pos = interactions[interactions['rating'] >= MIN_RATING].copy()
    print(f"    positive edges: {len(pos):,}")

    # ── 3. User / Food 필터 ───────────────────────────────────────────────────
    print(f"\n[3] Filtering users (>={MIN_USER_INTER}) & foods (>={MIN_FOOD_INTER})...")
    # 반복적으로 필터링 (수렴할 때까지)
    for i in range(5):
        uc = pos.groupby('user_id')['recipe_id'].count()
        pos = pos[pos['user_id'].isin(uc[uc >= MIN_USER_INTER].index)]
        fc = pos.groupby('recipe_id')['user_id'].count()
        pos = pos[pos['recipe_id'].isin(fc[fc >= MIN_FOOD_INTER].index)]
    print(f"    after filter: {len(pos):,} edges")
    print(f"    users:  {pos['user_id'].nunique():,}")
    print(f"    foods:  {pos['recipe_id'].nunique():,}")

    # ── 4. ID 매핑 ────────────────────────────────────────────────────────────
    print("\n[4] Building ID mappings...")
    user_ids = sorted(pos['user_id'].unique())
    food_ids = sorted(pos['recipe_id'].unique())
    user2idx = {u: i for i, u in enumerate(user_ids)}
    food2idx = {f: i for i, f in enumerate(food_ids)}
    N_USERS = len(user_ids)
    N_FOODS = len(food_ids)
    print(f"    N_users={N_USERS:,}  N_foods={N_FOODS:,}")

    # ── 5. Recipe 데이터 필터 및 파싱 ────────────────────────────────────────
    print("\n[5] Parsing recipe features...")
    recs = recipes[recipes['id'].isin(set(food_ids))].copy()
    recs['ing_list']  = recs['ingredients'].apply(parse_list_col)
    recs['nutr_list'] = recs['nutrition'].apply(parse_list_col)
    recs['health_score'] = recs['nutr_list'].apply(compute_health_score)
    recs = recs.set_index('id')

    # ── 6. Ingredient 매핑 ────────────────────────────────────────────────────
    print(f"\n[6] Building ingredient mapping (top {MAX_ING})...")
    ing_counter = defaultdict(int)
    for fid in food_ids:
        if fid in recs.index:
            for ing in recs.loc[fid, 'ing_list']:
                ing_counter[ing] += 1

    top_ings = sorted(ing_counter, key=lambda x: -ing_counter[x])[:MAX_ING]
    ing2idx  = {ing: i for i, ing in enumerate(top_ings)}
    N_INGS   = len(ing2idx)
    print(f"    N_ingredients={N_INGS:,}")

    # ── 7. 엣지 구성 ─────────────────────────────────────────────────────────
    print("\n[7] Building edges...")

    # user-eats-food
    src_u = [user2idx[u] for u in pos['user_id']]
    dst_f = [food2idx[f] for f in pos['recipe_id']]
    eats_ei = torch.tensor([src_u, dst_f], dtype=torch.long)

    # food-contains-ingredient
    cont_src, cont_dst = [], []
    for fid in food_ids:
        if fid in recs.index:
            f_idx = food2idx[fid]
            for ing in recs.loc[fid, 'ing_list']:
                if ing in ing2idx:
                    cont_src.append(f_idx)
                    cont_dst.append(ing2idx[ing])
    contains_ei = torch.tensor([cont_src, cont_dst], dtype=torch.long)

    # user-healthness-food (health score as edge_attr)
    health_scores_food = torch.zeros(N_FOODS, dtype=torch.float)
    for fid in food_ids:
        f_idx = food2idx[fid]
        if fid in recs.index:
            health_scores_food[f_idx] = float(recs.loc[fid, 'health_score'])

    # healthness edges = same as eats edges, but with health score attr
    health_attr = health_scores_food[torch.tensor(dst_f)]

    print(f"    eats edges:     {eats_ei.shape[1]:,}")
    print(f"    contains edges: {contains_ei.shape[1]:,}")

    # food-similar-food (top-5 Jaccard)
    print(f"\n[8] Computing food-food similarity (top-{MAX_SIMILAR} Jaccard, sampled)...")
    # 메모리 절약: food당 ingredient set 구성
    food_ing_sets = {}
    for fid in food_ids:
        if fid in recs.index:
            s = set(ing for ing in recs.loc[fid, 'ing_list'] if ing in ing2idx)
            food_ing_sets[food2idx[fid]] = s

    # 전체 Jaccard는 너무 느리므로 ingredient별로 인덱싱
    ing_to_foods = defaultdict(list)
    for f_idx, s in food_ing_sets.items():
        for ing in s:
            if ing in ing2idx:
                ing_to_foods[ing2idx[ing]].append(f_idx)

    sim_src, sim_dst = [], []
    processed = 0
    for f_idx, s in food_ing_sets.items():
        if not s:
            continue
        # 같은 ingredient를 공유하는 food들만 비교 (candidate)
        cands = defaultdict(int)
        for ing in s:
            if ing in ing2idx:
                for f2 in ing_to_foods[ing2idx[ing]]:
                    if f2 != f_idx:
                        cands[f2] += 1
        # Jaccard 계산 (상위 후보만)
        top_cands = sorted(cands, key=lambda x: -cands[x])[:50]
        scores = []
        for f2 in top_cands:
            s2 = food_ing_sets.get(f2, set())
            if not s2:
                continue
            j = len(s & s2) / len(s | s2)
            scores.append((f2, j))
        scores.sort(key=lambda x: -x[1])
        for f2, _ in scores[:MAX_SIMILAR]:
            sim_src.append(f_idx)
            sim_dst.append(f2)
        processed += 1
        if processed % 5000 == 0:
            print(f"    processed {processed}/{N_FOODS} foods...")

    similar_ei = torch.tensor([sim_src, sim_dst], dtype=torch.long)
    print(f"    similar edges:  {similar_ei.shape[1]:,}")

    # ── 9. Node Features ─────────────────────────────────────────────────────
    print("\n[9] Building node features...")

    # User features: interaction count, avg rating, std rating, recency + padding
    user_feats = np.zeros((N_USERS, USER_FEAT_DIM), dtype=np.float32)
    for uid, u_idx in user2idx.items():
        u_rows = pos[pos['user_id'] == uid]
        cnt    = len(u_rows)
        avg_r  = u_rows['rating'].mean()
        std_r  = u_rows['rating'].std() if cnt > 1 else 0.0
        user_feats[u_idx, 0] = np.log1p(cnt)
        user_feats[u_idx, 1] = avg_r / 5.0
        user_feats[u_idx, 2] = std_r / 5.0

    # Food features: nutrition (7) + n_ingredients(1) + n_steps(1) + health(1) + pad
    food_feats = np.zeros((N_FOODS, FOOD_FEAT_DIM), dtype=np.float32)
    for fid in food_ids:
        f_idx = food2idx[fid]
        if fid not in recs.index:
            continue
        row = recs.loc[fid]
        nl  = row['nutr_list']
        if len(nl) >= 7:
            # Normalize: calories/1000, rest /100
            food_feats[f_idx, 0]  = min(float(nl[0]) / 1000.0, 1.0)
            food_feats[f_idx, 1:7] = [min(float(v)/100.0, 1.0) for v in nl[1:7]]
        food_feats[f_idx, 7]  = min(float(row.get('n_ingredients', 0)) / 30.0, 1.0)
        food_feats[f_idx, 8]  = min(float(row.get('n_steps', 0)) / 20.0, 1.0)
        food_feats[f_idx, 9]  = float(row['health_score'])

    # Ingredient features: frequency + embedding dim padding
    ing_feats = np.zeros((N_INGS, 8), dtype=np.float32)
    for ing, i_idx in ing2idx.items():
        ing_feats[i_idx, 0] = np.log1p(ing_counter[ing]) / np.log1p(max(ing_counter.values()))

    # Dummy time node (required by NutriGraphNet schema)
    N_TIMES = 12  # months
    time_feats = np.eye(N_TIMES, dtype=np.float32)

    print(f"    user_feats:       {user_feats.shape}")
    print(f"    food_feats:       {food_feats.shape}")
    print(f"    ingredient_feats: {ing_feats.shape}")

    # ── 10. HeteroData 구성 ───────────────────────────────────────────────────
    print("\n[10] Assembling HeteroData...")
    hdata = HeteroData()

    hdata['user'].x        = torch.from_numpy(user_feats)
    hdata['food'].x        = torch.from_numpy(food_feats)
    hdata['ingredient'].x  = torch.from_numpy(ing_feats)
    hdata['time'].x        = torch.from_numpy(time_feats)

    # node count 명시
    hdata['user'].num_nodes       = N_USERS
    hdata['food'].num_nodes       = N_FOODS
    hdata['ingredient'].num_nodes = N_INGS
    hdata['time'].num_nodes       = N_TIMES

    # Health score 저장 (HealthGain 계산용)
    hdata['food'].health_score = health_scores_food

    # Edges
    hdata[('user','eats','food')].edge_index       = eats_ei
    hdata[('food','rev_eats','user')].edge_index   = eats_ei.flip(0)
    hdata[('user','healthness','food')].edge_index = eats_ei
    hdata[('user','healthness','food')].edge_attr  = health_attr.unsqueeze(-1)
    hdata[('food','rev_healthness','user')].edge_index = eats_ei.flip(0)
    hdata[('food','contains','ingredient')].edge_index     = contains_ei
    hdata[('ingredient','rev_contains','food')].edge_index = contains_ei.flip(0)
    hdata[('food','similar','food')].edge_index    = similar_ei

    # ── 11. 저장 ─────────────────────────────────────────────────────────────
    print(f"\n[11] Saving to {OUT_F}...")
    with open(OUT_F, 'wb') as f:
        pickle.dump(hdata, f)

    # ── 12. 요약 ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ✅ Food.com Graph Summary")
    print("="*60)
    print(f"  Users        : {N_USERS:,}  (feat={USER_FEAT_DIM})")
    print(f"  Foods        : {N_FOODS:,}  (feat={FOOD_FEAT_DIM})")
    print(f"  Ingredients  : {N_INGS:,}")
    print(f"  Interactions : {eats_ei.shape[1]:,}")
    print(f"  Contains     : {contains_ei.shape[1]:,}")
    print(f"  Similar      : {similar_ei.shape[1]:,}")
    hs_mean = health_scores_food.mean().item()
    print(f"  Health score (avg): {hs_mean:.4f}")
    print(f"\n  Saved: {OUT_F}")

    # user/food ratio 확인 (HFRS-DA 비교용)
    print(f"\n  [vs HFRS-DA Allrecipes] Users≈25K / Foods≈16K")
    print(f"  [This dataset]          Users={N_USERS:,} / Foods={N_FOODS:,}")
    return hdata


if __name__ == '__main__':
    main()
