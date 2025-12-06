"""
Personalized Health Score Calculator
논문의 개인화된 건강 점수 계산 방식 구현
"""

import torch
import numpy as np

class PersonalizedHealthScoreCalculator:
    """
    논문의 Personalized Health Score Calculation 구현
    
    사용자의 개인 특성(나이, 성별, 신장, 체중, 활동량)을 기반으로
    개인화된 영양 기준을 계산하고, 음식의 건강 점수를 평가합니다.
    """
    
    def __init__(self):
        # EER 계산 계수 (성별 및 연령대별)
        self.eer_coefficients = {
            'male_adult': {'a': 662, 'b': -9.53, 'c': 15.91, 'd': 539.6, 'e': 0},
            'female_adult': {'a': 354, 'b': -6.91, 'c': 9.36, 'd': 726, 'e': 0},
            # 필요시 다른 연령대 추가 가능
        }
        
        # 활동 계수 (Physical Activity coefficient)
        self.pa_coefficients = {
            'sedentary': {'male': 1.0, 'female': 1.0},
            'low_active': {'male': 1.11, 'female': 1.12},
            'active': {'male': 1.25, 'female': 1.27},
            'very_active': {'male': 1.48, 'female': 1.45}
        }
        
        # 영양소 권장 비율 (EER 대비 %)
        self.nutrient_standards = {
            'protein': {'min': 0.07, 'max': 0.20},  # 7-20% of energy
            'fat': {'min': 0.15, 'max': 0.30},      # 15-30% of energy
            'fiber': 12,  # g per 1000 kcal
            'cholesterol': 300,  # mg daily
            'sugar': 0.10,  # max 10% of energy
            'saturated_fat': 0.10,  # max 10% of fat intake
            'sodium': 1500  # mg daily (adjusted by energy)
        }
    
    def calculate_eer(self, age, gender, height, weight, activity_level):
        """
        Estimated Energy Requirement (EER) 계산
        
        Args:
            age: 나이
            gender: 성별 ('male' or 'female')
            height: 신장 (cm)
            weight: 체중 (kg)
            activity_level: 활동량 ('sedentary', 'low_active', 'active', 'very_active')
            
        Returns:
            float: 계산된 EER (kcal/day)
        """
        # 연령대 결정 (간단히 성인으로 처리)
        age_group = 'adult'
        coef_key = f"{gender}_{age_group}"
        
        if coef_key not in self.eer_coefficients:
            coef_key = 'male_adult'  # 기본값
        
        coef = self.eer_coefficients[coef_key]
        
        # PA 계수
        if activity_level not in self.pa_coefficients:
            activity_level = 'low_active'  # 기본값
        
        pa = self.pa_coefficients[activity_level][gender]
        
        # EER 계산
        eer = (coef['a'] + 
               coef['b'] * age + 
               pa * (coef['c'] * weight + coef['d'] * height) + 
               coef['e'])
        
        return max(eer, 1200)  # 최소 1200 kcal
    
    def calculate_personalized_standards(self, eer):
        """
        개인화된 영양 기준 계산
        
        Args:
            eer: Estimated Energy Requirement
            
        Returns:
            dict: 영양소별 개인화된 기준값
        """
        standards = {}
        
        # 단백질 (7-20% of energy, 1g = 4kcal)
        standards['protein'] = {
            'min': eer * self.nutrient_standards['protein']['min'] / 4,
            'max': eer * self.nutrient_standards['protein']['max'] / 4
        }
        
        # 지방 (15-30% of energy, 1g = 9kcal)
        standards['fat'] = {
            'min': eer * self.nutrient_standards['fat']['min'] / 9,
            'max': eer * self.nutrient_standards['fat']['max'] / 9
        }
        
        # 식이섬유 (12g per 1000 kcal)
        standards['fiber'] = self.nutrient_standards['fiber'] * (eer / 1000)
        
        # 콜레스테롤 (고정값)
        standards['cholesterol'] = self.nutrient_standards['cholesterol']
        
        # 당류 (max 10% of energy, 1g = 4kcal)
        standards['sugar'] = eer * self.nutrient_standards['sugar'] / 4
        
        # 포화지방 (max 10% of fat intake)
        standards['saturated_fat'] = standards['fat']['max'] * self.nutrient_standards['saturated_fat']
        
        # 나트륨 (조정된 값)
        standards['sodium'] = self.nutrient_standards['sodium'] * (eer / 2000)
        
        return standards
    
    def calculate_food_health_score(self, food_nutrients, personalized_standards):
        """
        음식의 개인화된 건강 점수 계산
        
        Args:
            food_nutrients: dict with keys: energy, protein, fat, fiber, cholesterol, 
                           sugar, saturated_fat, sodium
            personalized_standards: dict from calculate_personalized_standards()
            
        Returns:
            float: 0-1 사이의 건강 점수
        """
        scores = []
        
        # 단백질 (유익한 영양소)
        if 'protein' in food_nutrients and personalized_standards['protein']['max'] > 0:
            protein_score = min(
                food_nutrients['protein'] / personalized_standards['protein']['max'], 
                2.0
            )
            scores.append(protein_score)
        
        # 식이섬유 (유익한 영양소)
        if 'fiber' in food_nutrients and personalized_standards['fiber'] > 0:
            fiber_score = min(
                food_nutrients['fiber'] / personalized_standards['fiber'], 
                2.0
            )
            scores.append(fiber_score)
        
        # 지방 (제한 영양소)
        if 'fat' in food_nutrients and personalized_standards['fat']['max'] > 0:
            fat_score = max(
                -food_nutrients['fat'] / personalized_standards['fat']['max'], 
                -2.0
            )
            scores.append(fat_score)
        
        # 콜레스테롤 (제한 영양소)
        if 'cholesterol' in food_nutrients and personalized_standards['cholesterol'] > 0:
            chol_score = max(
                -food_nutrients['cholesterol'] / personalized_standards['cholesterol'], 
                -2.0
            )
            scores.append(chol_score)
        
        # 당류 (제한 영양소)
        if 'sugar' in food_nutrients and personalized_standards['sugar'] > 0:
            sugar_score = max(
                -food_nutrients['sugar'] / personalized_standards['sugar'], 
                -2.0
            )
            scores.append(sugar_score)
        
        # 포화지방 (제한 영양소)
        if 'saturated_fat' in food_nutrients and personalized_standards['saturated_fat'] > 0:
            sat_fat_score = max(
                -food_nutrients['saturated_fat'] / personalized_standards['saturated_fat'], 
                -2.0
            )
            scores.append(sat_fat_score)
        
        # 나트륨 (제한 영양소)
        if 'sodium' in food_nutrients and personalized_standards['sodium'] > 0:
            sodium_score = max(
                -food_nutrients['sodium'] / personalized_standards['sodium'], 
                -2.0
            )
            scores.append(sodium_score)
        
        # 평균 점수 계산
        if len(scores) == 0:
            return 0.5  # 기본값
        
        raw_score = sum(scores) / len(scores)
        
        # Min-max 정규화 (0-1 범위로)
        # raw_score 범위: -2 ~ 2
        normalized_score = (raw_score + 2) / 4
        
        return max(0.0, min(1.0, normalized_score))
    
    def calculate_batch_health_scores(self, user_features, food_features):
        """
        배치 단위로 건강 점수 계산
        
        Args:
            user_features: tensor of shape (n_users, user_feature_dim)
                         Expected features: [age, gender, height, weight, activity, ...]
            food_features: tensor of shape (n_foods, food_feature_dim)
                          Expected features: [energy, carbs, protein, fat, sat_fat, 
                                             cholesterol, fiber, sugar, sodium, ...]
            
        Returns:
            torch.Tensor: Health scores matrix (n_users, n_foods)
        """
        device = user_features.device
        n_users = user_features.shape[0]
        n_foods = food_features.shape[0]
        
        # Health scores matrix 초기화
        health_scores = torch.zeros((n_users, n_foods), device=device)
        
        # 각 사용자별로 EER 및 건강 점수 계산
        for user_idx in range(n_users):
            user_data = user_features[user_idx].cpu().numpy()
            
            # 사용자 특성 추출 (인덱스는 데이터에 따라 조정 필요)
            # 예: [age, gender_encoded, height, weight, activity_encoded, ...]
            age = user_data[0] if len(user_data) > 0 else 30
            gender = 'male' if user_data[1] > 0.5 else 'female' if len(user_data) > 1 else 'male'
            height = user_data[2] if len(user_data) > 2 else 170
            weight = user_data[3] if len(user_data) > 3 else 70
            activity = 'low_active'  # 기본값 (필요시 인코딩된 값에서 추출)
            
            # EER 계산
            eer = self.calculate_eer(age, gender, height, weight, activity)
            
            # 개인화된 영양 기준
            standards = self.calculate_personalized_standards(eer)
            
            # 각 음식의 건강 점수 계산
            for food_idx in range(n_foods):
                food_data = food_features[food_idx].cpu().numpy()
                
                # 음식 영양소 추출
                food_nutrients = {
                    'energy': food_data[0] if len(food_data) > 0 else 0,
                    'carbs': food_data[1] if len(food_data) > 1 else 0,
                    'protein': food_data[2] if len(food_data) > 2 else 0,
                    'fat': food_data[3] if len(food_data) > 3 else 0,
                    'saturated_fat': food_data[4] if len(food_data) > 4 else 0,
                    'cholesterol': food_data[5] if len(food_data) > 5 else 0,
                    'fiber': food_data[6] if len(food_data) > 6 else 0,
                    'sugar': food_data[7] if len(food_data) > 7 else 0,
                    'sodium': food_data[8] if len(food_data) > 8 else 0,
                }
                
                # 건강 점수 계산
                score = self.calculate_food_health_score(food_nutrients, standards)
                health_scores[user_idx, food_idx] = score
        
        return health_scores


def precompute_health_scores_for_dataset(data, calculator=None):
    """
    데이터셋 전체에 대한 건강 점수 사전 계산
    
    Args:
        data: HeteroData object with user and food features
        calculator: PersonalizedHealthScoreCalculator instance
        
    Returns:
        dict: {'edge_index': tensor, 'edge_attr': tensor}
    """
    if calculator is None:
        calculator = PersonalizedHealthScoreCalculator()
    
    print("Computing personalized health scores for all user-food pairs...")
    
    # User와 Food 특성 추출
    user_features = data.x_dict['user']
    food_features = data.x_dict['food']
    
    # 건강 점수 계산 (샘플링으로 메모리 절약)
    # 전체 계산은 메모리 부담이 크므로, 실제 엣지만 계산
    
    if ('user', 'eats', 'food') in data.edge_index_dict:
        edge_index = data[('user', 'eats', 'food')].edge_index
        user_indices = edge_index[0].cpu().numpy()
        food_indices = edge_index[1].cpu().numpy()
        
        n_edges = len(user_indices)
        health_scores = torch.zeros(n_edges, device=user_features.device)
        
        print(f"Computing health scores for {n_edges:,} edges...")
        
        # 배치 처리로 효율성 향상
        batch_size = 10000
        for start_idx in range(0, n_edges, batch_size):
            end_idx = min(start_idx + batch_size, n_edges)
            
            batch_user_idx = user_indices[start_idx:end_idx]
            batch_food_idx = food_indices[start_idx:end_idx]
            
            # 배치별 계산
            for i, (u_idx, f_idx) in enumerate(zip(batch_user_idx, batch_food_idx)):
                user_data = user_features[u_idx].cpu().numpy()
                food_data = food_features[f_idx].cpu().numpy()
                
                # 간소화된 계산 (성능을 위해)
                age = user_data[0] if len(user_data) > 0 else 30
                gender = 'male' if user_data[1] > 0.5 else 'female' if len(user_data) > 1 else 'male'
                height = user_data[2] if len(user_data) > 2 else 170
                weight = user_data[3] if len(user_data) > 3 else 70
                
                eer = calculator.calculate_eer(age, gender, height, weight, 'low_active')
                standards = calculator.calculate_personalized_standards(eer)
                
                food_nutrients = {
                    'energy': food_data[0] if len(food_data) > 0 else 0,
                    'protein': food_data[2] if len(food_data) > 2 else 0,
                    'fat': food_data[3] if len(food_data) > 3 else 0,
                    'saturated_fat': food_data[4] if len(food_data) > 4 else 0,
                    'cholesterol': food_data[5] if len(food_data) > 5 else 0,
                    'fiber': food_data[6] if len(food_data) > 6 else 0,
                    'sugar': food_data[7] if len(food_data) > 7 else 0,
                    'sodium': food_data[8] if len(food_data) > 8 else 0,
                }
                
                score = calculator.calculate_food_health_score(food_nutrients, standards)
                health_scores[start_idx + i] = score
            
            if (start_idx // batch_size) % 10 == 0:
                print(f"  Progress: {end_idx:,}/{n_edges:,} ({100*end_idx/n_edges:.1f}%)")
        
        print(f"✅ Health scores computed! Range: [{health_scores.min():.3f}, {health_scores.max():.3f}]")
        
        return {
            'edge_index': edge_index,
            'edge_attr': health_scores
        }
    
    return None
