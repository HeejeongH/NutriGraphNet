# 🪟 Windows 실행 가이드 (최종 수정본)

## ⚠️ 중요: 데이터 문제 해결됨!

**문제:** Log1p 정규화로 인해 학습이 안 되는 문제  
**해결:** 역정규화 후 MinMax 재정규화 적용 (`processed_data_GNN_v4.pkl`)

---

## 1️⃣ 코드 업데이트

```bash
# Anaconda Prompt 또는 터미널 열기
conda activate cuda

# 프로젝트 폴더로 이동
cd "C:\Users\user\OneDrive\Heejeong\식의학유전체실\과제\진행\# 맞춤형 식이설계 플랫폼\#공공플랫폼\#기업 맞춤형 서비스\알고리즘\GNN_RecSys"

# 최신 코드 받기
git pull origin main
```

---

## 2️⃣ 데이터 재생성 (중요!)

```bash
# 역정규화 후 재정규화
python reverse_normalization.py
```

**출력 확인:**
```
✅ Applying MinMax normalization...
📊 Final (MinMax normalized) Stats:
   Min: 0.000000
   Max: 1.000000
   Mean: 0.004992    ← ✅ 낮은 값 (이전: 0.109)
   Median: 0.003467  ← ✅ 낮은 값 (이전: 0.097)
✅ Saved! File size: 38.52 MB
```

---

## 3️⃣ 빠른 테스트 (10 epochs)

```bash
python train_v2.py --data_path data/processed_data/processed_data_GNN_v4.pkl --model graphsage --epochs 10 --hidden_channels 128 --out_channels 64
```

**확인 사항:**
- ✅ Loss가 감소하는지 (50.83 → 48.xx → 46.xx ...)
- ✅ F1 Score가 개선되는지 (0.66 → 0.70+ 목표)
- ✅ AUC가 0.5 이상인지 (랜덤 추측보다 나아야 함)

**예상 결과:**
```
Epoch  5/10 | Train Loss: 48.25 | Val F1: 0.68 | Val AUC: 0.63
Epoch 10/10 | Train Loss: 46.50 | Val F1: 0.72 | Val AUC: 0.68
```

---

## 4️⃣ 전체 실험 (50 epochs)

### A. 실험 스크립트 생성

```bash
python src/run_health_aware_experiments.py --epochs 50
```

### B. 실험 스크립트 수정 (필수!)

`run_health_experiments.bat` 파일을 열고 **모든 `processed_data_GNN_fixed.pkl`을 `processed_data_GNN_v4.pkl`로 수정**:

```batch
REM 수정 전
python train_v2.py --data_path data/processed_data/processed_data_GNN_fixed.pkl ...

REM 수정 후
python train_v2.py --data_path data/processed_data/processed_data_GNN_v4.pkl ...
```

### C. 실험 실행

```bash
# Windows CMD 또는 Anaconda Prompt에서
run_health_experiments.bat
```

**또는 Git Bash에서:**
```bash
bash run_health_experiments.sh
```

**6개 실험 구성:**
1. ✅ Vanilla GNN (Baseline)
2. ✅ GraphSAGE (Baseline)
3. ✅ GraphSAGE + Health Loss
4. ✅ NutriGraphNet V2 (Full) - Health Attention + Health Loss
5. ✅ NutriGraphNet V2 - Health Attention Only
6. ✅ NutriGraphNet V2 - Health Loss Only

**예상 시간:** 3-5시간 (GPU 사용 시)

---

## 5️⃣ 결과 확인

```bash
python src/compare_health_results.py
```

**생성되는 파일:**
- `results/health_experiments/preference_metrics_comparison.png` - 선호도 메트릭
- `results/health_experiments/health_metrics_comparison.png` - 건강 메트릭
- `results/health_experiments/radar_comparison.png` - 레이더 차트
- `results/health_experiments/topk_metrics_comparison.png` - Top-K 메트릭
- `results/health_experiments/experiment_report.txt` - 상세 보고서
- `results/health_experiments/comparison_results.csv` - 결과 CSV

---

## 🚨 문제 해결

### 문제 1: `conda activate cuda` 실패
```bash
# Anaconda Prompt를 관리자 권한으로 실행
conda init cmd.exe
```

### 문제 2: Git 인증 오류
```bash
git config --global credential.helper manager
```

### 문제 3: 메모리 부족
```bash
# hidden_channels 줄이기
python train_v2.py --data_path data/processed_data/processed_data_GNN_v4.pkl --model graphsage --epochs 10 --hidden_channels 64 --out_channels 32
```

### 문제 4: CUDA 오류
```bash
# CPU로 실행
python train_v2.py --data_path data/processed_data/processed_data_GNN_v4.pkl --model graphsage --epochs 10 --hidden_channels 128 --out_channels 64 --device cpu
```

---

## 📊 예상 결과

### Baseline Models (Preference Only)
- **Vanilla GNN**: F1 0.70, AUC 0.65
- **GraphSAGE**: F1 0.72, AUC 0.68

### Health-aware Models
- **GraphSAGE + Health Loss**: F1 0.73, AUC 0.70, Health Score ↑
- **NutriGraphNet V2 (Full)**: F1 0.75, AUC 0.72, Health Score ↑↑

### 건강 메트릭 개선 목표
- Avg Health Score: 0.55 → **0.68** (+24%)
- Health Precision: 0.40 → **0.62** (+55%)
- Health-aware F1: 0.60 → **0.72** (+20%)

---

## ✅ 체크리스트

- [ ] 1단계: 코드 업데이트 (`git pull origin main`)
- [ ] 2단계: 데이터 재생성 (`python reverse_normalization.py`)
- [ ] 3단계: 빠른 테스트 (10 epochs)
  - [ ] Loss 감소 확인
  - [ ] F1 > 0.70 확인
  - [ ] AUC > 0.60 확인
- [ ] 4단계: 배치 파일 수정 (`processed_data_GNN_v4.pkl` 경로)
- [ ] 5단계: 전체 실험 실행 (50 epochs)
- [ ] 6단계: 결과 비교 (`python src/compare_health_results.py`)

---

## 🎯 핵심 변경사항

| 항목 | 이전 (v3) | 현재 (v4) | 개선 |
|------|----------|----------|-----|
| **정규화 방식** | Log1p | MinMax | ✅ |
| **Mean** | 0.109 | 0.005 | ✅ 95% 감소 |
| **Median** | 0.097 | 0.003 | ✅ 97% 감소 |
| **학습 가능성** | ❌ (Loss 고정) | ✅ (Loss 감소) | ✅ |
| **AUC** | 0.50 (랜덤) | 0.65+ (학습) | ✅ |

---

## 📞 지원

문제가 발생하면:
1. 터미널 출력 전체를 캡처
2. `results/health_experiments/` 폴더 내용 확인
3. 로그 파일 공유

**GitHub:** https://github.com/HeejeongH/NutriGraphNet

---

**마지막 업데이트:** 2024-01-XX  
**작성자:** NutriGraphNet Team
