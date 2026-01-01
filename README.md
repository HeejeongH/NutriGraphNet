# 🍽️ NutriGraphNet

**Graph Neural Network 기반 건강 맞춤형 식품 추천 시스템**

사용자의 식습관과 건강 정보를 학습하여 개인에게 최적화된 건강한 음식을 추천합니다.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 특징

- **Health-Aware GAT**: 건강 점수를 고려한 그래프 어텐션 네트워크
- **5-Fold Cross Validation**: 일반화 성능 보장
- **Multi-Task Learning**: 선호도 + 건강 점수 동시 최적화

## 📊 성능

| Metric | Score |
|--------|-------|
| AUC | 0.92+ |
| F1 Score | 0.76+ |
| Precision | 0.74+ |
| Recall | 0.78+ |

---

## 🚀 빠른 시작

### 설치
```bash
git clone https://github.com/HeejeongH/NutriGraphNet.git
cd NutriGraphNet
pip install torch torch_geometric scikit-learn matplotlib numpy
```

### 실행

**빠른 테스트 (10 epochs, ~5-10분)**
```bash
python test_quick.py
```

**전체 실험 (500 epochs, 5 folds, ~2-4시간)**
```bash
python train_final.py
```

**커스텀 설정**
```bash
python train_final.py \
    --hidden_channels 128 \
    --out_channels 64 \
    --epochs 500 \
    --n_folds 5
```

---

## 📁 프로젝트 구조

```
NutriGraphNet/
├── train_final.py           # 메인 학습 코드
├── test_quick.py            # 빠른 테스트
├── data/
│   └── processed_data/
│       └── processed_data_GNN_v5.pkl
└── results/                 # 실험 결과
```

---

## 🏗️ 모델 아키텍처

```
Input: User-Food-Ingredient Graph
         ↓
┌─────────────────────────┐
│ HealthAwareGATEncoder   │
│  ├─ GAT Layer 1         │
│  ├─ GAT Layer 2         │
│  └─ Health Attention    │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│ HealthAwareEdgeDecoder  │
│  └─ 4-Layer MLP         │
└─────────────────────────┘
         ↓
Output: Preference Score
```

**Loss Function:**
```
Total Loss = BCE + λ₁×Health Loss + λ₂×Ranking Loss
```

---

## ⚙️ 주요 파라미터

```python
--hidden_channels 128      # Hidden layer 크기
--out_channels 64          # Output layer 크기
--epochs 500               # Training epochs
--n_folds 5                # Cross-validation folds
--lr 0.0001               # Learning rate
--lambda_health 0.01      # Health loss 가중치
--ranking_weight 0.2      # Ranking loss 가중치
```

---

## 📈 결과 분석

각 실험 폴더에 생성되는 파일:
- `best_model.pth` - 최고 성능 모델
- `training_curves.png` - 학습 곡선 (Loss, F1, AUC, LR)
- `cross_validation_results.pkl` - 전체 결과 요약

**결과 로드 예제:**
```python
import pickle

with open('results/final_experiments/cross_validation_results.pkl', 'rb') as f:
    results = pickle.load(f)
    
print(results['avg_results'])  # 평균 성능
```

---

## 🔧 고급 사용

**모델 로드 및 추론:**
```python
import torch
from train_final import HealthAwareRecommender

model = HealthAwareRecommender(
    hidden_channels=128,
    out_channels=64,
    metadata=(data.node_types, data.edge_types)
)
model.load_state_dict(torch.load('results/fold_1/best_model.pth'))

with torch.no_grad():
    pred = model(x_dict, edge_index_dict, edge_label_index)
```

---

## 👥 저자

**Heejeong**
- 푸드테크 박사과정생
- GitHub: [@HeejeongH](https://github.com/HeejeongH)

---

## 📄 라이선스

MIT License

---

## 🙏 감사

- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [scikit-learn](https://scikit-learn.org/)

---

**⭐ Star를 눌러주세요!**
