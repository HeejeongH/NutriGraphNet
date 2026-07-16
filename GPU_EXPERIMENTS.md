# GPU 실험 실행 가이드 — NutriGraphNet v2
**대상 실험: EXP-A, EXP-D, EXP-F** (EXP-B/C/G는 GPU 완료)

---

## 0. 준비 사항

```bash
# NutriGraphNet 폴더로 이동
cd /path/to/NutriGraphNet

# 필수 파일 확인
ls data/processed_data/processed_data_GNN_v5.pkl   # 데이터
ls nutrigraphnet_v2.py                              # 모델
ls run_analysis_experiments.py                      # 실험 스크립트

# GPU 확인
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 결과 폴더 생성
mkdir -p results/gpu
```

---

## EXP-A: SGL Augmentation Ratio Sweep

**목적:** edge dropout ratio p∈{0.0~0.5}에서 SGL이 어떻게 붕괴하는가  
**소요:** ~2시간 (GPU, 5-fold × 6 ratio)

### 방법 1: 스크립트 자동 실행 (권장)
```bash
python src/run_analysis_experiments.py --exp A
```

### 방법 2: 수동 실행
```bash
for AUG in 0.0 0.1 0.2 0.3 0.4 0.5; do
    python src/nutrigraphnet_v2.py \
        --variants sgl \
        --sgl_aug $AUG \
        --n_folds 5 \
        --epochs 300 \
        --patience 30 \
        --hidden_channels 128 \
        --out_channels 64 \
        --num_layers 3 \
        --heads 4 \
        --seed 42 \
        --output_dir results/gpu/A_sgl_aug_${AUG}
    echo "✓ Done: sgl_aug=$AUG"
done
```

### 예상 결과 폴더
```
results/gpu/A_sgl_aug_0.0/all_results.json
results/gpu/A_sgl_aug_0.1/all_results.json
results/gpu/A_sgl_aug_0.2/all_results.json
results/gpu/A_sgl_aug_0.3/all_results.json
results/gpu/A_sgl_aug_0.4/all_results.json
results/gpu/A_sgl_aug_0.5/all_results.json
```

---

## EXP-D: Embedding Dimension Sweep

**목적:** dim∈{16,32,64,128,256}에서 모든 모델의 capacity 특성 분석  
**소요:** ~5시간 (GPU, 5모델 × 5 dim × 3-fold)

### 방법 1: 스크립트 자동 실행 (권장)
```bash
python src/run_analysis_experiments.py --exp D
```

### 방법 2: 수동 실행
```bash
MODELS="mf,lightgcn,ngcf,sgl,hfrsda"

for DIM in 16 32 64 128 256; do
    python src/nutrigraphnet_v2.py \
        --variants $MODELS \
        --out_channels $DIM \
        --hidden_channels $((DIM * 2)) \
        --n_folds 3 \
        --epochs 200 \
        --patience 30 \
        --seed 42 \
        --output_dir results/gpu/D_dim_${DIM}
    echo "✓ Done: dim=$DIM"
done
```

### 예상 결과 폴더
```
results/gpu/D_dim_16/all_results.json    # mf, lightgcn, ngcf, sgl, hfrsda
results/gpu/D_dim_32/all_results.json
results/gpu/D_dim_64/all_results.json
results/gpu/D_dim_128/all_results.json
results/gpu/D_dim_256/all_results.json
```

---

## EXP-F: Graph Component Ablation

**목적:** 각 auxiliary edge type 제거 시 NutriGraphNet 성능 변화 측정  
**소요:** ~4시간 (GPU, 7 variant × 5-fold)

> ⚠️ **중요:** EXP-F v1(HFRS-DA)은 이미 CPU에서 완료됨 (topology invariance 확인).  
> GPU에서는 **EXP-F v2: NutriGraphNet ablation** 을 실행합니다.

### 방법 1: 스크립트 자동 실행 (권장)
```bash
python src/run_analysis_experiments.py --exp F
```

### 방법 2: 수동 실행 (NutriGraphNet ablation만)
```bash
BASE_ARGS="--variants full --n_folds 5 --epochs 300 --patience 30
           --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42"

# Full graph (베이스라인)
python src/nutrigraphnet_v2.py $BASE_ARGS \
    --output_dir results/gpu/F_ablation_full_graph

# w/o ingredient edges
python src/nutrigraphnet_v2.py $BASE_ARGS \
    --ablate_no_ingredient \
    --output_dir results/gpu/F_ablation_no_ingredient

# w/o time edges
python src/nutrigraphnet_v2.py $BASE_ARGS \
    --ablate_no_time \
    --output_dir results/gpu/F_ablation_no_time

# w/o food-similar edges
python src/nutrigraphnet_v2.py $BASE_ARGS \
    --ablate_no_food_similar \
    --output_dir results/gpu/F_ablation_no_food_similar

# w/o healthness edges
python src/nutrigraphnet_v2.py $BASE_ARGS \
    --ablate_no_healthness \
    --output_dir results/gpu/F_ablation_no_healthness

# w/o ingredient + time
python src/nutrigraphnet_v2.py $BASE_ARGS \
    --ablate_no_ingredient --ablate_no_time \
    --output_dir results/gpu/F_ablation_no_ingredient_time

# w/o all auxiliary (ingredient + time + food_similar)
python src/nutrigraphnet_v2.py $BASE_ARGS \
    --ablate_no_ingredient --ablate_no_time --ablate_no_food_similar \
    --output_dir results/gpu/F_ablation_no_all_auxiliary
```

### 예상 결과 폴더
```
results/gpu/F_ablation_full_graph/all_results.json
results/gpu/F_ablation_no_ingredient/all_results.json
results/gpu/F_ablation_no_time/all_results.json
results/gpu/F_ablation_no_food_similar/all_results.json
results/gpu/F_ablation_no_healthness/all_results.json
results/gpu/F_ablation_no_ingredient_time/all_results.json
results/gpu/F_ablation_no_all_auxiliary/all_results.json
```

---

## 전체 한 번에 실행 (A + D + F)

```bash
# 순차 실행 (총 ~11시간, 하룻밤 돌리기)
python src/run_analysis_experiments.py --exp A && \
python src/run_analysis_experiments.py --exp D && \
python src/run_analysis_experiments.py --exp F

# 또는 백그라운드 실행 (로그 저장)
nohup bash -c "
  python src/run_analysis_experiments.py --exp A > logs/expA.log 2>&1 && echo 'EXP-A DONE' &&
  python src/run_analysis_experiments.py --exp D > logs/expD.log 2>&1 && echo 'EXP-D DONE' &&
  python src/run_analysis_experiments.py --exp F > logs/expF.log 2>&1 && echo 'EXP-F DONE'
" &
echo "PID: $!"
```

---

## 실험 완료 후: SUMMARY.json 갱신

실험이 끝나면 results/gpu/ 폴더에 새 폴더들이 생깁니다.  
아래 명령으로 SUMMARY 파일을 한 번에 갱신하세요:

```bash
python src/generate_summary_gpu.py results/gpu/
```

생성 파일:
- `results/gpu/SUMMARY.json`       ← 논문용 구조화 JSON
- `results/gpu/SUMMARY_flat.json`  ← flat 포맷 (업로드용)
- `results/gpu/SUMMARY_readable.txt`

---

## 빠른 검증 (실험 전 확인용, ~5분)

```bash
# Quick mode: 1-fold, 30 epochs으로 파이프라인 정상 동작 확인
python src/run_analysis_experiments.py --exp A --quick
python src/run_analysis_experiments.py --exp D --quick
python src/run_analysis_experiments.py --exp F --quick
```

---

## VRAM별 권장 파라미터

| VRAM | hidden | out | layers | heads | 비고 |
|------|--------|-----|--------|-------|------|
| 4GB  | 64 | 32 | 1 | 2 | OOM 안전 |
| 8GB  | 128 | 64 | 3 | 4 | **기본값 (권장)** |
| 16GB+ | 256 | 128 | 3 | 8 | 고성능 |

4GB VRAM 환경이라면 수동 실행 시 아래 추가:
```bash
--hidden_channels 64 --out_channels 32 --num_layers 1 --heads 2
```
