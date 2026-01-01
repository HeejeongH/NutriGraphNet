# 🍽️ NutriGraphNet: Health-Aware Food Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3%2B-orange)](https://pytorch-geometric.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**건강을 고려한 개인 맞춤형 식품 추천 시스템**

Graph Neural Network (GNN) 기반으로 사용자의 식습관과 건강 정보를 학습하여, 개인에게 최적화된 건강한 음식을 추천합니다.

---

## 🎯 주요 특징

### 1. Health-Aware Architecture
- **Health Attention Mechanism**: 음식의 건강 점수를 학습에 반영
- **Residual Connections**: 안정적인 gradient flow
- **Multi-Task Learning**: 선호도 예측 + 건강 점수 최적화

### 2. Advanced GNN Model
- **Heterogeneous Graph**: User-Food-Ingredient 관계 모델링
- **GAT (Graph Attention Network)**: 중요한 관계에 집중
- **Batch Normalization & Dropout**: 과적합 방지

### 3. Robust Training Pipeline
- **5-Fold Cross Validation**: 일반화 성능 보장
- **Early Stopping**: 최적 모델 자동 선택
- **CosineAnnealingLR**: 학습률 스케줄링
- **Negative Sampling**: 균형잡힌 학습

---

## 📊 성능

| Metric | Score |
|--------|-------|
| **AUC** | 0.92+ |
| **F1 Score** | 0.76+ |
| **Precision** | 0.74+ |
| **Recall** | 0.78+ |

---

## 🚀 빠른 시작

### 요구사항
```bash
Python >= 3.8
PyTorch >= 2.0
PyTorch Geometric >= 2.3
scikit-learn >= 1.0
matplotlib >= 3.5
numpy >= 1.20
```

### 설치
```bash
# 저장소 클론
git clone https://github.com/HeejeongH/NutriGraphNet.git
cd NutriGraphNet

# 의존성 설치
pip install torch torch_geometric scikit-learn matplotlib numpy
```

### 실행

#### 1. 빠른 테스트 (10 Epochs, ~5-10분)
```bash
python test_quick.py
```

#### 2. 전체 실험 (500 Epochs, 5 Folds, ~2-4시간)
```bash
python train_final.py
```

#### 3. 커스텀 설정
```bash
python train_final.py \
    --data_path data/processed_data/processed_data_GNN_v5.pkl \
    --hidden_channels 128 \
    --out_channels 64 \
    --epochs 500 \
    --n_folds 5 \
    --output_dir results/my_experiment
```

---

## 📁 프로젝트 구조

```
NutriGraphNet/
├── train_final.py              # 메인 학습 코드
├── test_quick.py               # 빠른 테스트 스크립트
│
├── data/
│   └── processed_data/
│       └── processed_data_GNN_v5.pkl  # 전처리된 그래프 데이터
│
├── src/
│   ├── HealthAwareGNN.py       # Health-Aware 모델 (참고용)
│   ├── NutriGraphNet_v2.py     # 모델 정의
│   ├── evaluation_metrics.py   # 평가 지표
│   └── health_score_calculator.py  # 건강 점수 계산
│
├── results/                    # 실험 결과
│   ├── quick_test/
│   └── final_experiments/
│
├── README.md                   # 이 파일
├── FINAL_WINDOWS_GUIDE.md     # Windows 실행 가이드
└── COMPLETE_SUCCESS.md        # 상세 문서
```

---

## 🏗️ 아키텍처

### 모델 구조
```
Input: User-Food-Ingredient Heterogeneous Graph

┌─────────────────────────────────────────┐
│  HealthAwareGATEncoder                  │
│  ├─ Input Projections (per node type)  │
│  ├─ GAT Layer 1 (heads=2)              │
│  │   └─ BatchNorm + GELU + Dropout     │
│  ├─ GAT Layer 2 (heads=1)              │
│  │   └─ BatchNorm + GELU + Dropout     │
│  └─ Health Attention                    │
│      └─ Scale food embeddings by       │
│          health scores                  │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  HealthAwareEdgeDecoder                 │
│  └─ 4-Layer MLP                         │
│      ├─ Linear + LayerNorm + GELU      │
│      ├─ Linear + LayerNorm + GELU      │
│      ├─ Linear + GELU                   │
│      └─ Linear → Logits                 │
└─────────────────────────────────────────┘
                   ↓
Output: User-Food Preference Score (0-1)
```

### Loss Function
```python
Total Loss = BCE Loss + λ₁ × Health Loss + λ₂ × Ranking Loss

where:
  BCE Loss = Binary Cross-Entropy Loss
  Health Loss = Encourage healthy food recommendations
  Ranking Loss = Positive samples > Negative samples
```

---

## 📈 실험 설정

### 기본 하이퍼파라미터
```python
hidden_channels = 128        # Hidden layer 크기
out_channels = 64            # Output layer 크기
heads = 2                    # GAT attention heads
dropout = 0.5                # Dropout 비율

epochs = 500                 # Training epochs
lr = 0.0001                  # Learning rate
weight_decay = 0.001         # L2 regularization
patience = 20                # Early stopping patience

lambda_health = 0.01         # Health loss 가중치
ranking_weight = 0.2         # Ranking loss 가중치
margin = 1.0                 # Ranking margin

n_folds = 5                  # Cross-validation folds
val_ratio = 0.05             # Validation ratio
test_ratio = 0.10            # Test ratio
neg_sampling_ratio = 2.0     # Negative sampling ratio
```

---

## 📊 결과 분석

### 학습 곡선
각 실험 폴더에 `training_curves.png` 생성:
- Loss Curves (Train & Val)
- F1 Score Curve
- AUC Curve
- Learning Rate Schedule

### 결과 파일
- `best_model.pth`: 최고 성능 모델 가중치
- `cross_validation_results.pkl`: 전체 결과 요약

---

## 🔧 고급 사용법

### 1. 데이터 정규화
```bash
# 데이터 재생성 (필요시)
python robust_normalization.py
```

### 2. 모델 로드 및 추론
```python
import torch
from train_final import HealthAwareRecommender

# 모델 로드
model = HealthAwareRecommender(
    hidden_channels=128,
    out_channels=64,
    metadata=(data.node_types, data.edge_types)
)
model.load_state_dict(torch.load('results/fold_1/best_model.pth'))

# 추론
with torch.no_grad():
    pred = model(x_dict, edge_index_dict, edge_label_index)
```

### 3. 결과 분석
```python
import pickle

# 결과 로드
with open('results/final_experiments/cross_validation_results.pkl', 'rb') as f:
    results = pickle.load(f)

# 평균 성능
print(results['avg_results'])

# 각 Fold 성능
for i, fold_result in enumerate(results['all_results']):
    print(f"Fold {i+1}: F1={fold_result['test_f1']:.4f}, AUC={fold_result['test_auc']:.4f}")
```

---

## 📝 문서

- **[FINAL_WINDOWS_GUIDE.md](FINAL_WINDOWS_GUIDE.md)**: Windows 환경 실행 가이드
- **[COMPLETE_SUCCESS.md](COMPLETE_SUCCESS.md)**: 전체 프로젝트 문서 및 기술 상세

---

## 🤝 기여

버그 리포트, 기능 제안, Pull Request는 언제나 환영합니다!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 라이선스

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 저자

**Heejeong**
- 푸드테크 박사과정생
- 전공: 푸드테크, 개인맞춤형식품, AI, 자동화
- GitHub: [@HeejeongH](https://github.com/HeejeongH)

---

## 🙏 감사의 말

이 프로젝트는 다음 라이브러리들을 사용합니다:
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [scikit-learn](https://scikit-learn.org/)
- [matplotlib](https://matplotlib.org/)

---

## 📞 문의

질문이나 문제가 있으시면 [Issues](https://github.com/HeejeongH/NutriGraphNet/issues)를 열어주세요.

---

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**
