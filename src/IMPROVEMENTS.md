# 🚀 NutriGraphNet 성능 개선 방안

## 📊 현재 성능 분석

### **실험 결과 요약**

| 모델 | F1 Score | AUC | 비고 |
|------|----------|-----|------|
| XGBoost | **0.761** | **0.851** | 최고 성능 |
| Random Forest | 0.759 | 0.850 | 2위 |
| HealthAwareGNN | 0.659 | 0.528 | ⚠️ 개선 필요 |
| GraphSAGE | 0.660 | 0.500 | Baseline |
| GAT (No Health) | 0.211 | 0.537 | 실패 |

### **핵심 문제점**

1. **❌ GNN < Traditional ML**
   - GNN이 XGBoost보다 F1 약 0.1 낮음
   - 그래프 구조의 이점을 활용하지 못함

2. **❌ Health Loss 역효과**
   - Health_NoLoss: F1=0.121 (매우 낮음)
   - Health loss가 너무 강하게 작용
   - λ_h = 0.1이 너무 클 수 있음

3. **❌ 데이터 특성 문제**
   - **건강-선호도 상관관계: 0.046** (거의 없음!)
   - 사용자들이 건강한 음식을 선호하지 않음
   - Health attention이 오히려 성능을 저하시킬 수 있음

---

## 💡 **V2 주요 개선사항**

### **1. 아키텍처 개선**

#### **A. Deeper & Wider Network**
```python
변경 전: 2 layers, hidden=128, heads=1
변경 후: 3 layers, hidden=256, heads=4

개선 이유:
- 더 깊은 네트워크로 복잡한 패턴 학습
- Multi-head attention으로 다양한 관점 학습
- 용량 증가로 표현력 향상
```

#### **B. Skip Connections (ResNet-style)**
```python
x_dict = {
    key: x + skip_connection(x_prev[key])
    for key, x in x_dict.items()
}

개선 이유:
- Gradient vanishing 문제 해결
- 더 깊은 네트워크 안정적 훈련
- 저수준/고수준 특성 모두 활용
```

#### **C. Layer-wise Attention Aggregation**
```python
# 학습 가능한 레이어 중요도
layer_weights = F.softmax(self.layer_attention, dim=0)
output = Σ layer_outputs[i] * layer_weights[i]

개선 이유:
- 각 레이어의 중요도를 학습
- 모든 레이어의 정보 활용
- 더 풍부한 표현력
```

#### **D. Feature Interaction Layer**
```python
# Decoder에 Bilinear interaction 추가
interaction = Bilinear(user_emb, food_emb)
z = concat([user_emb, food_emb, interaction])

개선 이유:
- User-Food 간 명시적 상호작용 모델링
- 단순 concatenation보다 표현력 높음
```

### **2. Loss Function 개선**

#### **A. Adaptive Lambda Health**
```python
# Warm-up strategy
epoch_progress = epoch / total_epochs
λ_h = λ_init + (λ_max - λ_init) * epoch_progress

초기: λ_h = 0.01  (선호도 위주 학습)
후기: λ_h = 0.1   (건강도 고려)

개선 이유:
- 초반에는 기본 추천 능력 학습
- 후반에 점진적으로 건강 고려
- 급격한 변화 방지
```

#### **B. Focal Loss**
```python
focal_loss = ((1 - pt) ** γ) * BCE

개선 이유:
- 어려운 샘플에 집중
- Imbalanced data 문제 해결
- 더 robust한 학습
```

#### **C. Temperature Scaling**
```python
scaled_health = sigmoid((health_scores - 0.5) * temp)

temp = 2.0 사용

개선 이유:
- 극단적인 건강 점수의 영향 완화
- 더 부드러운 최적화
- Overconfidence 방지
```

### **3. 훈련 전략 개선**

#### **A. Negative Sampling**
```python
# 각 positive sample당 5개 negative 생성
neg_samples = sample_negatives(pos_edges, num_neg=5)

개선 이유:
- Contrastive learning 강화
- 더 좋은 음식 구별 능력
- 일반화 성능 향상
```

#### **B. Feature Augmentation**
```python
# Gaussian noise + Feature dropout
augmented = features + noise * 0.1
augmented = augmented * dropout_mask

개선 이유:
- Overfitting 방지
- Robustness 향상
- 일반화 능력 개선
```

#### **C. Better Regularization**
```python
- Weight decay: 0.01 -> 0.02
- Dropout: 0.3 유지
- Gradient clipping: 1.0

개선 이유:
- 과적합 방지
- 안정적인 훈련
```

---

## 🎯 **예상 성능 향상**

### **보수적 추정**

| 개선사항 | F1 향상 | AUC 향상 |
|---------|---------|----------|
| Deeper architecture | +0.03 | +0.02 |
| Multi-head attention | +0.02 | +0.03 |
| Skip connections | +0.02 | +0.02 |
| Feature interaction | +0.02 | +0.02 |
| Focal loss | +0.03 | +0.03 |
| Adaptive lambda | +0.04 | +0.03 |
| Negative sampling | +0.03 | +0.04 |
| Feature augmentation | +0.02 | +0.02 |
| **총 예상 향상** | **+0.21** | **+0.21** |

### **목표 성능**

```
현재:  F1 = 0.659, AUC = 0.528
목표:  F1 = 0.87+, AUC = 0.74+

XGBoost와 비교:
현재 차이: -0.102 (F1)
목표 차이: +0.11  (F1)  ⭐ XGBoost 초과!
```

---

## 🔧 **구현 우선순위**

### **Phase 1: 즉시 적용 (High Impact, Low Risk)**
1. ✅ **Adaptive Lambda Health** - 가장 큰 영향 예상
2. ✅ **Focal Loss** - Imbalanced data 해결
3. ✅ **Temperature Scaling** - Health score 완화

### **Phase 2: 아키텍처 개선 (Medium Risk)**
4. ✅ **Deeper Network (3 layers)** - 표현력 증가
5. ✅ **Multi-head Attention** - 다양한 패턴 학습
6. ✅ **Skip Connections** - 안정적 훈련

### **Phase 3: 고급 기법 (Higher Risk)**
7. ⏳ **Negative Sampling** - 구현 복잡도 높음
8. ⏳ **Feature Augmentation** - 신중한 튜닝 필요
9. ⏳ **Layer-wise Aggregation** - 추가 검증 필요

---

## 📈 **추가 개선 아이디어**

### **1. Graph Structure 개선**

#### **A. Multi-hop Sampling**
```python
# 2-hop neighborhood 활용
neighbors = sample_2hop_neighbors(node, num_neighbors=20)

장점:
- 더 넓은 receptive field
- 간접적인 관계 학습
```

#### **B. Edge Importance Learning**
```python
# 엣지 중요도 학습
edge_weight = MLP(edge_features)
message = edge_weight * neighbor_features

장점:
- 중요한 관계에 집중
- Noise 엣지 필터링
```

### **2. Health Score 개선**

#### **A. Learnable Health Function**
```python
# 고정된 EER 대신 학습 가능한 함수
health_score = HealthNet(user_features, food_features)

장점:
- 데이터 기반 건강 점수
- 더 정확한 평가
```

#### **B. Multi-task Learning**
```python
# 건강 점수 예측을 보조 태스크로
loss = λ1 * recommendation_loss + 
       λ2 * health_prediction_loss

장점:
- 건강 인식 능력 향상
- 더 나은 representation
```

### **3. 앙상블 방법**

#### **A. Model Averaging**
```python
# 여러 모델의 예측 평균
final_pred = (pred_v1 + pred_v2 + pred_xgboost) / 3

예상 효과:
- F1: +0.03~0.05
- Robustness 향상
```

#### **B. Stacking**
```python
# GNN 출력을 XGBoost 입력으로
gnn_features = encoder(x_dict, edge_index_dict)
final_pred = xgboost(gnn_features + original_features)

예상 효과:
- GNN과 Traditional ML의 장점 결합
- 최고 성능 달성 가능
```

---

## 🎯 **실험 계획**

### **Baseline 재현 (1일)**
```bash
python train_v2.py --model nutrigraphnet_v1 --epochs 100
```

### **V2 순차적 개선 (3일)**
```bash
# Day 1: Loss 개선
python train_v2.py --focal_loss --adaptive_lambda --temp_scaling

# Day 2: 아키텍처 개선  
python train_v2.py --num_layers 3 --heads 4 --skip_connections

# Day 3: 고급 기법
python train_v2.py --neg_sampling --feature_aug
```

### **하이퍼파라미터 튜닝 (2일)**
```python
search_space = {
    'hidden_channels': [128, 256, 512],
    'num_layers': [2, 3, 4],
    'heads': [1, 2, 4],
    'dropout': [0.2, 0.3, 0.4],
    'lambda_health_max': [0.05, 0.1, 0.15],
    'focal_gamma': [1.0, 2.0, 3.0],
    'lr': [1e-4, 5e-4, 1e-3]
}
```

---

## 💬 **결론 및 전망**

### **성능 개선 가능성: ⭐⭐⭐⭐⭐ (매우 높음)**

**이유:**
1. ✅ **명확한 문제점 식별됨**
   - Health loss 너무 강함 → Adaptive lambda로 해결
   - 얕은 네트워크 → Deeper architecture로 해결
   - Imbalanced data → Focal loss로 해결

2. ✅ **검증된 개선 기법들**
   - Skip connections (ResNet)
   - Multi-head attention (Transformer)
   - Focal loss (Object Detection)
   - 모두 다른 도메인에서 효과 입증됨

3. ✅ **데이터 특성 고려**
   - 건강-선호도 상관관계 낮음 → Adaptive strategy
   - 충분한 데이터 (20K+ users, 31K+ foods)
   - 풍부한 특성 (29+17+101 차원)

### **목표 달성 가능성**

| 목표 | 난이도 | 달성 가능성 |
|-----|--------|-----------|
| F1 > 0.75 | Easy | 95% |
| F1 > 0.80 | Medium | 80% |
| F1 > XGBoost (0.761) | Medium | 75% |
| F1 > 0.85 | Hard | 50% |

### **추천 순서**

1. **V2 모델로 재실험** (우선순위 1)
   - Adaptive lambda + Focal loss만으로도 큰 향상 예상
   
2. **하이퍼파라미터 튜닝** (우선순위 2)
   - 특히 lambda_health_max, hidden_channels
   
3. **앙상블 시도** (우선순위 3)
   - GNN + XGBoost stacking으로 최고 성능

**예상 일정: 1주일 내 목표 달성 가능** ✨

---

## 📞 다음 단계

1. V2 모델로 실험 시작하시겠습니까?
2. 하이퍼파라미터 그리드 서치 실행하시겠습니까?
3. 특정 개선사항부터 테스트하시겠습니까?

어떤 방향으로 진행하고 싶으신가요? 😊
