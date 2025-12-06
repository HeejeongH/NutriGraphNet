"""
Health-aware 평가 메트릭
연구 목적: 선호도 예측 + 건강도 고려를 모두 평가
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def compute_comprehensive_metrics(predictions, targets, health_scores, 
                                   pred_threshold=0.5, health_threshold=0.6):
    """
    종합 평가 메트릭
    
    Args:
        predictions: 모델의 추천 확률 (0-1)
        targets: 실제 선호도 레이블 (0 or 1)
        health_scores: 음식의 건강 점수 (0-1)
        pred_threshold: 추천 판단 threshold
        health_threshold: 건강식 판단 threshold
        
    Returns:
        dict: 선호도 + 건강도 메트릭
    """
    
    # Numpy 변환
    predictions = predictions.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    health_scores = health_scores.cpu().detach().numpy()
    
    # Binary predictions
    pred_binary = (predictions > pred_threshold).astype(int)
    
    # ============================================
    # 1. 기본 선호도 예측 메트릭
    # ============================================
    preference_metrics = {
        'accuracy': accuracy_score(targets, pred_binary),
        'precision': precision_score(targets, pred_binary, zero_division=0),
        'recall': recall_score(targets, pred_binary, zero_division=0),
        'f1': f1_score(targets, pred_binary, zero_division=0),
        'auc': roc_auc_score(targets, predictions) if len(np.unique(targets)) > 1 else 0.5
    }
    
    # ============================================
    # 2. 건강도 고려 메트릭 (핵심!)
    # ============================================
    
    # 추천된 음식들의 평균 건강 점수
    recommended_indices = np.where(pred_binary == 1)[0]
    
    if len(recommended_indices) > 0:
        avg_health_of_recommendations = health_scores[recommended_indices].mean()
        
        # 건강식 추천 비율 (추천된 것 중 건강식 비율)
        healthy_food_mask = health_scores > health_threshold
        healthy_recommendations = np.sum(pred_binary * healthy_food_mask)
        health_precision = healthy_recommendations / len(recommended_indices)
    else:
        avg_health_of_recommendations = 0.0
        health_precision = 0.0
    
    # 실제로 선호하는 음식들의 건강 점수
    preferred_indices = np.where(targets == 1)[0]
    if len(preferred_indices) > 0:
        avg_health_of_preferences = health_scores[preferred_indices].mean()
    else:
        avg_health_of_preferences = 0.0
    
    health_metrics = {
        'avg_health_score': avg_health_of_recommendations,  # 추천 음식의 평균 건강도
        'health_precision': health_precision,  # 건강식 추천 정밀도
        'health_improvement': avg_health_of_recommendations - avg_health_of_preferences  # 건강도 향상
    }
    
    # ============================================
    # 3. 선호도-건강도 균형 메트릭
    # ============================================
    
    # F1 score와 Health score의 조화 평균 (Health-aware F1)
    if preference_metrics['f1'] > 0 and health_metrics['avg_health_score'] > 0:
        health_aware_f1 = 2 * (preference_metrics['f1'] * health_metrics['avg_health_score']) / \
                         (preference_metrics['f1'] + health_metrics['avg_health_score'])
    else:
        health_aware_f1 = 0.0
    
    # Top-K 추천의 건강도 (상위 10개 추천)
    top_k = min(10, len(predictions))
    top_k_indices = np.argsort(predictions)[-top_k:]
    top_k_health = health_scores[top_k_indices].mean()
    
    # Top-K의 정확도
    top_k_accuracy = targets[top_k_indices].mean()
    
    balance_metrics = {
        'health_aware_f1': health_aware_f1,  # 선호도-건강도 조화 메트릭
        'top_k_health': top_k_health,  # Top-K 추천의 건강도
        'top_k_accuracy': top_k_accuracy  # Top-K 추천의 정확도
    }
    
    # ============================================
    # 4. Health-aware Recall (중요!)
    # ============================================
    # "건강한 음식 중에서 얼마나 많이 추천했는가?"
    healthy_food_indices = np.where(health_scores > health_threshold)[0]
    
    if len(healthy_food_indices) > 0:
        healthy_recommendations_count = np.sum(pred_binary[healthy_food_indices])
        health_aware_recall = healthy_recommendations_count / len(healthy_food_indices)
    else:
        health_aware_recall = 0.0
    
    health_metrics['health_aware_recall'] = health_aware_recall
    
    # 통합 결과
    return {
        **preference_metrics,
        **health_metrics,
        **balance_metrics
    }


def print_metrics_comparison(baseline_metrics, health_aware_metrics, model_names=None):
    """
    Baseline vs Health-aware 모델 비교 출력
    
    Args:
        baseline_metrics: Baseline 모델 메트릭 dict
        health_aware_metrics: Health-aware 모델 메트릭 dict
        model_names: (baseline_name, health_aware_name) tuple
    """
    
    if model_names is None:
        model_names = ("Baseline", "Health-aware")
    
    baseline_name, health_name = model_names
    
    print(f"\n{'='*80}")
    print(f"📊 Model Comparison: {baseline_name} vs {health_name}")
    print(f"{'='*80}\n")
    
    # 선호도 예측 메트릭
    print("1️⃣ Preference Prediction Metrics:")
    print(f"{'Metric':<20} {baseline_name:>15} {health_name:>15} {'Δ':>10}")
    print("-" * 65)
    
    preference_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    for key in preference_keys:
        baseline_val = baseline_metrics.get(key, 0)
        health_val = health_aware_metrics.get(key, 0)
        delta = health_val - baseline_val
        delta_str = f"{delta:+.4f}"
        
        print(f"{key.upper():<20} {baseline_val:>15.4f} {health_val:>15.4f} {delta_str:>10}")
    
    # 건강도 메트릭
    print(f"\n2️⃣ Health-awareness Metrics:")
    print(f"{'Metric':<20} {baseline_name:>15} {health_name:>15} {'Δ':>10}")
    print("-" * 65)
    
    health_keys = ['avg_health_score', 'health_precision', 'health_improvement', 
                   'health_aware_recall']
    for key in health_keys:
        baseline_val = baseline_metrics.get(key, 0)
        health_val = health_aware_metrics.get(key, 0)
        delta = health_val - baseline_val
        delta_str = f"{delta:+.4f}"
        
        print(f"{key:<20} {baseline_val:>15.4f} {health_val:>15.4f} {delta_str:>10}")
    
    # 균형 메트릭
    print(f"\n3️⃣ Balance Metrics:")
    print(f"{'Metric':<20} {baseline_name:>15} {health_name:>15} {'Δ':>10}")
    print("-" * 65)
    
    balance_keys = ['health_aware_f1', 'top_k_health', 'top_k_accuracy']
    for key in balance_keys:
        baseline_val = baseline_metrics.get(key, 0)
        health_val = health_aware_metrics.get(key, 0)
        delta = health_val - baseline_val
        delta_str = f"{delta:+.4f}"
        
        print(f"{key:<20} {baseline_val:>15.4f} {health_val:>15.4f} {delta_str:>10}")
    
    print(f"\n{'='*80}\n")
    
    # 핵심 결과 요약
    print("📌 Key Findings:")
    
    # F1 Score 비교
    f1_delta = health_aware_metrics['f1'] - baseline_metrics['f1']
    if f1_delta > 0.01:
        print(f"   ✅ F1 Score improved by {f1_delta:.2%}")
    elif f1_delta < -0.01:
        print(f"   ⚠️  F1 Score decreased by {abs(f1_delta):.2%}")
    else:
        print(f"   ➡️  F1 Score maintained ({f1_delta:+.2%})")
    
    # Health Score 비교
    health_delta = health_aware_metrics['avg_health_score'] - baseline_metrics['avg_health_score']
    if health_delta > 0.05:
        print(f"   ✅ Average health score improved by {health_delta:.2%}")
    elif health_delta < -0.05:
        print(f"   ⚠️  Average health score decreased by {abs(health_delta):.2%}")
    else:
        print(f"   ➡️  Average health score maintained ({health_delta:+.2%})")
    
    # Health-aware F1
    ha_f1_baseline = baseline_metrics['health_aware_f1']
    ha_f1_health = health_aware_metrics['health_aware_f1']
    
    if ha_f1_health > ha_f1_baseline * 1.05:
        print(f"   ✅ Health-aware F1 improved by {(ha_f1_health/ha_f1_baseline - 1):.2%}")
    
    print(f"\n{'='*80}\n")


def compute_health_aware_ranking_metrics(predictions, targets, health_scores, k_list=[5, 10, 20]):
    """
    Ranking 기반 건강도 메트릭
    
    Args:
        predictions: 모델의 추천 확률
        targets: 실제 선호도
        health_scores: 건강 점수
        k_list: Top-K 값 리스트
        
    Returns:
        dict: Ranking 메트릭
    """
    
    predictions = predictions.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    health_scores = health_scores.cpu().detach().numpy()
    
    # 예측 점수 기준으로 내림차순 정렬
    sorted_indices = np.argsort(predictions)[::-1]
    
    metrics = {}
    
    for k in k_list:
        k = min(k, len(predictions))
        top_k_indices = sorted_indices[:k]
        
        # Top-K의 정확도
        top_k_accuracy = targets[top_k_indices].mean()
        
        # Top-K의 평균 건강 점수
        top_k_health = health_scores[top_k_indices].mean()
        
        # NDCG@K (health score를 relevance로 사용)
        dcg = np.sum((2 ** health_scores[top_k_indices] - 1) / np.log2(np.arange(2, k + 2)))
        
        # Ideal DCG (건강 점수 기준 정렬)
        ideal_indices = np.argsort(health_scores)[::-1][:k]
        idcg = np.sum((2 ** health_scores[ideal_indices] - 1) / np.log2(np.arange(2, k + 2)))
        
        ndcg = dcg / idcg if idcg > 0 else 0
        
        metrics[f'accuracy@{k}'] = top_k_accuracy
        metrics[f'health@{k}'] = top_k_health
        metrics[f'ndcg@{k}'] = ndcg
    
    return metrics


if __name__ == '__main__':
    # 테스트 예제
    print("Testing evaluation metrics...")
    
    # 더미 데이터
    torch.manual_seed(42)
    predictions = torch.rand(100)
    targets = torch.randint(0, 2, (100,)).float()
    health_scores = torch.rand(100) * 0.5 + 0.3  # 0.3-0.8 범위
    
    # 메트릭 계산
    metrics = compute_comprehensive_metrics(predictions, targets, health_scores)
    
    print("\n📊 Computed Metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")
    
    # Ranking 메트릭
    ranking_metrics = compute_health_aware_ranking_metrics(predictions, targets, health_scores)
    
    print("\n📈 Ranking Metrics:")
    for key, value in ranking_metrics.items():
        print(f"   {key}: {value:.4f}")
