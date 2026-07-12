# GPU 실행 가이드 — NutriGraphNet v2

**Updated:** 2026-07-12  
**Model:** NutriGraphNet v2 (버그 수정 3개 적용)  
**Auto GPU detection:** `torch.cuda.is_available()` → `--device` 인수 없이 자동 감지

---

## 0. 요구사항

```bash
# Python 패키지 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.x.x+cu121.html

# 또는 CPU 환경
pip install -r requirements.txt
```

| 항목 | CPU 환경 (sandbox) | GPU 환경 (권장) |
|------|-------------------|-----------------|
| VRAM | — | 8GB+ (전체 파라미터) / 4GB (경량 파라미터) |
| hidden_channels | 64 (OOM 방지) | 128 (기본값) |
| out_channels | 32 (OOM 방지) | 64 (기본값) |
| num_layers | 1 (OOM 방지) | 3 (기본값) |
| heads | 2 (OOM 방지) | 4 (기본값) |
| n_folds | 1 | 5 (5-Fold CV) |

---

## 1. 단일 실험 실행 (GPU)

### 기본 모델 비교 (5-fold CV)
```bash
# NutriGraphNet full model — GPU 기본값
python nutrigraphnet_v2.py \
    --variants full \
    --n_folds 5 \
    --epochs 300 \
    --patience 30 \
    --output_dir results/gpu/full_5fold

# HFRSDA 비교
python nutrigraphnet_v2.py \
    --variants hfrsda \
    --n_folds 5 \
    --epochs 300 \
    --patience 30 \
    --output_dir results/gpu/hfrsda_5fold

# 모든 모델 동시 실행 (GPU VRAM 16GB+)
python nutrigraphnet_v2.py \
    --variants mf,lightgcn,ngcf,sgl,hfrsda,full \
    --n_folds 5 \
    --epochs 300 \
    --patience 30 \
    --output_dir results/gpu/all_models_5fold
```

---

## 2. EXP-C: λ_health Sweep — NutriGraphNet (GPU 권장)

### GPU 환경 (전체 파라미터, 5-fold, 3-seed)
```bash
# GPU에서는 전체 파라미터 사용 가능
for lam in 0.0 0.001 0.005 0.01 0.05 0.1 0.5 1.0; do
    for seed in 42 123 777; do
        python nutrigraphnet_v2.py \
            --variants full \
            --lambda_health $lam \
            --n_folds 5 \
            --epochs 300 \
            --patience 30 \
            --hidden_channels 128 \
            --out_channels 64 \
            --num_layers 3 \
            --heads 4 \
            --seed $seed \
            --output_dir results/gpu/C_lambda_${lam}_s${seed}
        echo "Done: lambda=$lam seed=$seed"
    done
done
```

### CPU 환경 (sandbox, 메모리 최적화, 1-fold, seed=42)
```bash
# Sandbox에서 실행된 현재 결과 (이미 완료)
python run_analysis_experiments.py --exp C
# → 메모리 최적화: hidden=64, out=32, layers=1, heads=2, 1-fold
```

### run_analysis_experiments.py로 한 번에 실행
```bash
# EXP-C only
python run_analysis_experiments.py --exp C

# 빠른 검증 (1 fold, 30 epochs)
python run_analysis_experiments.py --exp C --quick
```

---

## 3. EXP-B: Sparsity Sweep (GPU 권장)

```bash
# 전 모델 (hfrsda 포함), 3-fold, 5개 sparsity
python run_analysis_experiments.py --exp B

# 또는 수동으로
for ratio in 0.1 0.3 0.5 0.7 1.0; do
    python nutrigraphnet_v2.py \
        --variants mf,lightgcn,ngcf,sgl,hfrsda \
        --interaction_ratio $ratio \
        --n_folds 3 \
        --epochs 300 \
        --patience 30 \
        --output_dir results/gpu/B_sparsity_${ratio}
done
```

---

## 4. EXP-G: Layer Depth Sweep (신규)

```bash
# LightGCN / NGCF — num_layers 1~4 (over-smoothing 분석)
python run_analysis_experiments.py --exp G

# 또는 수동으로
for nl in 1 2 3 4; do
    python nutrigraphnet_v2.py \
        --variants lightgcn,ngcf \
        --num_layers $nl \
        --n_folds 5 \
        --epochs 300 \
        --patience 30 \
        --output_dir results/gpu/G_layers_${nl}
done
```

---

## 5. EXP-D: Embedding Dimension Sweep

```bash
# 전 모델, dim=16~256
python run_analysis_experiments.py --exp D

# 또는 수동으로
for d in 16 32 64 128 256; do
    python nutrigraphnet_v2.py \
        --variants mf,lightgcn,ngcf,sgl,hfrsda \
        --out_channels $d \
        --hidden_channels $((d * 2)) \
        --n_folds 3 \
        --epochs 200 \
        --patience 30 \
        --output_dir results/gpu/D_dim_${d}
done
```

---

## 6. 전체 실험 파이프라인

```bash
# 전체 실험 (A+B+C+D+F+G) — GPU 환경에서 수 시간 소요
python run_analysis_experiments.py --exp all

# 결과 요약 출력
python run_analysis_experiments.py --exp summary
```

---

## 7. 현재 Sandbox 실험 결과 (CPU, 1-fold, seed=42)

### EXP-C NutriGraphNet λ_health Sweep 결과
| λ_health | AUC    | F1     | HR@10  | NDCG@10 | MRR    | health_loss |
|---------|--------|--------|--------|---------|--------|-------------|
| 0.000   | 0.8353 | 0.6560 | 0.6720 | 0.3515  | 0.2665 | 0.0000      |
| 0.001   | 0.8377 | 0.6558 | 0.6740 | 0.3823  | 0.3050 | 0.5050      |
| 0.005   | 0.8400 | 0.6560 | 0.6720 | 0.3254  | 0.2340 | 0.3403      |
| 0.010   | 0.8373 | 0.6561 | 0.6720 | 0.3448  | 0.2585 | 0.3822      |
| 0.050   | 0.8394 | 0.6557 | 0.6740 | 0.3858  | 0.3093 | 0.4434      |
| 0.100   | 0.8381 | 0.6558 | 0.6760 | 0.3923  | 0.3167 | 0.4088      |
| **0.500** | **0.8401** | 0.6526 | **0.6960** | 0.3923  | 0.3096 | 0.1259 ← **BEST** |
| 1.000   | 0.8099 | 0.6647 | 0.6900 | 0.4500  | 0.3865 | 0.0963      |

**Key finding:** λ=0.5가 HR@10=0.6960으로 최적 (+3.6% vs λ=0.0).  
Health loss가 실제로 작동함 (NutriGraphNet 버그 수정 후).

---

## 8. GPU에서 재현성 향상 (권장 설정)

```bash
# GPU 환경 최적 파라미터 (VRAM 8GB 기준)
python nutrigraphnet_v2.py \
    --variants full \
    --lambda_health 0.5 \
    --n_folds 5 \
    --epochs 300 \
    --patience 30 \
    --hidden_channels 128 \
    --out_channels 64 \
    --num_layers 3 \
    --heads 4 \
    --seed 42 \
    --output_dir results/gpu/C_best_lambda_0.5
```

GPU에서는 `num_layers=3, heads=4, hidden=128, out=64`로 더 강력한 모델 학습 가능.  
Sandbox 결과(1-fold, 경량)보다 수치가 향상될 것으로 예상.

---

## 9. 주의사항

1. **GPU 자동 감지**: `nutrigraphnet_v2.py`는 `--device` 인수가 없으며, `torch.cuda.is_available()`로 자동 감지.
2. **CUDA/CPU 수치 차이**: GPU 부동소수점 연산 결과는 CPU와 수치적으로 약간 다를 수 있음 (정상).
3. **OOM 방지**: VRAM 4GB 이하라면 `--hidden_channels 64 --out_channels 32 --num_layers 1 --heads 2` 사용.
4. **재현성**: `--seed 42` 고정. 단, 멀티-GPU 환경에서는 non-determinism 주의.
5. **데이터 경로**: `data/processed_data/processed_data_GNN_v5.pkl` 파일 필요.
