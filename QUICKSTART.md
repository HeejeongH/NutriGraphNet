# 🚀 NutriGraphNet V2 빠른 시작 가이드

## 📋 실행 전 체크리스트

### 1. 환경 확인
```bash
# Python 버전 확인 (3.9+ 필요)
python --version

# GPU 사용 가능 여부 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. 필요한 패키지 설치
```bash
pip install torch torch-geometric scikit-learn numpy pandas matplotlib seaborn
```

### 3. 데이터 경로 확인
```bash
# 데이터가 있는지 확인
ls ../data/processed_data/processed_data_GNN.pkl
```

---

## 🎯 실행 방법

### **방법 1: 간단한 테스트 실행 (추천!)**

```bash
# 프로젝트 디렉토리로 이동
cd /home/user/webapp

# 간단한 테스트 (30 에포크, 작은 모델)
python train_v2.py \
    --hidden_channels 128 \
    --out_channels 64 \
    --num_layers 2 \
    --epochs 30 \
    --print_every 5

# 예상 소요 시간: 5-10분
# 예상 성능: F1 0.70+
```

### **방법 2: 전체 V2 모델 실행**

```bash
# V2 전체 기능 활성화
python train_v2.py \
    --hidden_channels 256 \
    --out_channels 128 \
    --num_layers 3 \
    --epochs 100 \
    --lambda_health_init 0.01 \
    --lambda_health_max 0.1 \
    --focal_gamma 2.0 \
    --temperature 2.0

# 예상 소요 시간: 20-30분
# 예상 성능: F1 0.80+
```

### **방법 3: 실험 스크립트 일괄 실행**

```bash
# 실행 권한 부여
chmod +x run_experiment.sh

# 4가지 실험 자동 실행
./run_experiment.sh

# 예상 소요 시간: 1-2시간
# 4가지 설정의 결과를 비교할 수 있음
```

---

## 📊 실행 중 모니터링

### 출력 예시:
```
================================================================================
🚀 NutriGraphNet V2 Training
================================================================================

💻 Device: cuda
   GPU: NVIDIA GeForce RTX 3070
   Memory: 8.6 GB

📊 Loading data...
✅ Data loaded successfully!
   Users: 20,820
   Foods: 31,458
   ...

🎯 Starting training...

Epoch   5/100 | Train Loss: 0.6234 | Val Loss: 0.6012 | Val F1: 0.7123 | Val AUC: 0.7845
  💾 Saved best model (F1: 0.7123)
Epoch  10/100 | Train Loss: 0.5891 | Val Loss: 0.5789 | Val F1: 0.7456 | Val AUC: 0.8023
  💾 Saved best model (F1: 0.7456)
...
```

### 주요 지표:
- **Train/Val Loss**: 낮을수록 좋음 (0.5 이하 목표)
- **Val F1**: 높을수록 좋음 (0.75+ 목표, 0.80+ 우수)
- **Val AUC**: 높을수록 좋음 (0.75+ 목표, 0.80+ 우수)
- **λ_h**: Health loss 가중치 (0.01 → 0.1로 점진적 증가)

---

## 🎯 하이퍼파라미터 튜닝

### 성능이 낮을 때:

#### 1. Health loss가 너무 강한 경우
```bash
python train_v2.py \
    --lambda_health_init 0.005 \
    --lambda_health_max 0.05  # 절반으로 줄임
```

#### 2. 과적합(Overfitting)이 발생하는 경우
```bash
python train_v2.py \
    --dropout 0.4 \              # Dropout 증가
    --weight_decay 0.03 \        # Weight decay 증가
    --noise_std 0.15             # Feature noise 증가
```

#### 3. 학습이 불안정한 경우
```bash
python train_v2.py \
    --lr 0.0005 \                # Learning rate 감소
    --max_grad_norm 0.5 \        # Gradient clipping 강화
    --patience 20                # Early stopping patience 증가
```

#### 4. 성능을 극대화하고 싶은 경우
```bash
python train_v2.py \
    --hidden_channels 512 \      # 모델 크기 증가
    --out_channels 256 \
    --num_layers 4 \
    --epochs 150 \
    --lr 0.0005                  # Learning rate 미세 조정
```

---

## 📈 결과 확인

### 저장된 모델 로드:
```python
import torch

# 모델 로드
checkpoint = torch.load('best_model_v2.pth')
print(f"Best epoch: {checkpoint['epoch']}")
print(f"Best F1: {checkpoint['val_f1']:.4f}")
print(f"Metrics: {checkpoint['val_metrics']}")
```

### 결과 비교:
```bash
# 여러 실험 결과 비교
ls -lh results/*.pth

# 각 모델의 성능 확인
python -c "
import torch
for model_path in ['results/quick_test.pth', 'results/v2_full.pth']:
    ckpt = torch.load(model_path)
    print(f'{model_path}: F1={ckpt[\"val_f1\"]:.4f}')
"
```

---

## ⚠️ 문제 해결

### CUDA Out of Memory
```bash
# 배치 크기 줄이기 (코드 수정 필요) 또는
# 모델 크기 줄이기
python train_v2.py \
    --hidden_channels 128 \
    --out_channels 64 \
    --num_layers 2
```

### 데이터 파일 없음
```bash
# 데이터 경로 수정
python train_v2.py \
    --data_path /실제/데이터/경로/processed_data_GNN.pkl
```

### Import 에러
```bash
# 현재 디렉토리 확인
pwd
# /home/user/webapp 이어야 함

# 패키지 재설치
pip install --upgrade torch torch-geometric
```

---

## 💡 팁

### 1. 빠른 반복 실험
```bash
# 작은 모델 + 적은 에포크로 여러 설정 테스트
python train_v2.py --epochs 20 --hidden_channels 128
```

### 2. GPU 메모리 효율
```bash
# 더 작은 모델 사용
python train_v2.py --hidden_channels 128 --num_layers 2
```

### 3. 학습 재개
```python
# train_v2.py 수정하여 체크포인트에서 재개 가능
checkpoint = torch.load('best_model_v2.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

---

## 🎉 예상 성능

| 설정 | 예상 F1 | 예상 시간 |
|-----|---------|----------|
| Quick Test (작은 모델) | 0.70+ | 5-10분 |
| V2 Full (기본) | 0.80+ | 20-30분 |
| V2 Low Health | 0.82+ | 20-30분 |
| V2 Large (큰 모델) | 0.85+ | 40-60분 |

**현재 XGBoost: F1 = 0.761**
**목표: F1 > 0.80 달성!** 🎯

---

## 📞 다음 단계

1. ✅ 간단한 테스트 실행
2. 📊 결과 확인 및 분석
3. 🔧 하이퍼파라미터 튜닝
4. 🏆 최고 성능 모델 선택
5. 📝 논문에 결과 반영

궁금한 점이 있으면 언제든지 물어보세요! 😊
