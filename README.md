# NutriGraphNet

**보조 영양 구조로 부족한 상호작용 데이터를 대체하는 건강 인식 식품 추천**

국민건강영양조사 기반 이종 그래프(NutriGraph-KR)에서 두 가지를 검증한다 —
(1) 식재료·음식 유사도·식사 시점 같은 **보조 구조가 희소한 상호작용 데이터를 대체**할 수 있는가,
(2) **건강 제약이 아키텍처 수준에서 라우팅**되어 실제로 작동하고 조절 가능한가.

- 논문 초안: `논문/draft/paper_draft.md` (v2.0)
- 재프레이밍 근거·자산 실사: `논문/draft/reframing_proposal.md`

---

## 핵심 결과 (GPU 5-fold CV, NutriGraph-KR)

| | |
|---|---|
| 10% 밀도 HR@10 | **0.738** — 6개 모델 중 1위 (NGCF 0.509 대비 **+45.0%**) |
| 밀도 불변성 | 0.738 → 0.729 (10%→100%, ratio **0.99×**) |
| 데이터 효율 | 상호작용 **1/10**로 NGCF 전체 데이터 성능의 **94.1%** 달성 |
| 그래프 의존성 | auxiliary 엣지 전체 제거 시 **−21.0%** (대조군은 Δ=0.000) |
| 건강 라우팅 | HealthGain@10 ≈ −0.010이 모든 λ에서 non-zero; λ≥0.5에서 제어 가능한 trade-off |
| **한계** | 데이터가 늘어도 개선 없음 → 50% 이상에서 NGCF에 역전 (0.784 vs 0.729) |
| **한계** | NDCG@10은 전 밀도에서 대조군에 열세 (디코더 문제, health 탓 아님) |

---

## 저장소 구조

```
src/                          코드
  nutrigraphnet_v2.py         논문의 모델 + 베이스라인(MF/LightGCN/NGCF/SGL) + DualAttn-TB 대조군
  nutrigraphnet_v3.py         v3 변형 (§8.4 아키텍처-밀도 교차의 근거)
  run_analysis_experiments.py 실험 드라이버 (--exp summary 등)
  generate_summary_gpu.py     results/gpu/*/all_results.json → SUMMARY.json
  generate_figures.py         figures/ 생성
  graph_builder.py            NutriGraph-KR 그래프 구축
  build_foodcom_graph.py      Food.com 그래프 구축 (보조 데이터셋)
scripts/                      GPU 실행 배치 (리포지토리 루트에서 실행)
data/
  processed_data/processed_data_GNN_v5.pkl   NutriGraph-KR — 논문의 데이터셋
  foodcom/processed_foodcom.pkl              Food.com — 보조, 현재 논문 미사용
results/
  gpu/                        논문 Table 1 / B / C-Full / F의 출처 ★
  analysis/                   논문 Table A / D / G / C(경량 multi-seed)의 출처 ★
  v3_e300_lr1e3/, v3_expB_*/  §8.4 v2-v3 비교
figures/                      논문 Figure 1–6
논문/draft/                   논문 초안
etc/                          조사 코드자료집·전처리 노트북 등 원천 자료
```

### ★ results/ 는 논문의 증거이며 재생성 비용이 크다

모든 논문 수치가 여기서 나온다. **`.gitignore`에 `results/` 같은 디렉토리 통째 제외 규칙을
절대 다시 넣지 말 것** — git은 제외된 디렉토리 안의 파일을 `!`로 되살릴 수 없어서, 예전에 있던
`!results/analysis/` 규칙이 작성된 날부터 무효였고 결과 JSON 333개가 한 달간 추적 밖에 있었다.
무거운 재생성 가능 산출물(`*.pth`, `*.pt`, `results/**/*.pdf`, `results/**/*.png`)만
확장자로 제외한다.

---

## 실행

모든 명령은 **리포지토리 루트**에서 실행한다.

```bash
pip install torch torch_geometric scikit-learn matplotlib numpy pandas

# 단일 실험 (NutriGraphNet, λ=0.01, 5-fold, full params)
python src/nutrigraphnet_v2.py --variants full --lambda_health 0.01 \
  --n_folds 5 --epochs 300 --patience 30 \
  --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 \
  --output_dir results/gpu/my_run

# 배치 (Windows GPU 머신)
scripts\run_expB_full.bat      # EXP-B: NutriGraphNet 밀도 스윕
scripts\run_v3_fair.bat        # v3 공정 예산 비교
scripts\run_v3_expB.bat        # v3 저밀도 프로브

# 요약 및 그림
python src/generate_summary_gpu.py
python src/generate_figures.py
```

### variant 이름 주의 — 과거 사고의 원인

| CLI 값 | 실제 모델 |
|---|---|
| `full` | **NutriGraphNet** (논문의 모델) |
| `mf` / `lightgcn` / `ngcf` / `sgl` | 공개 베이스라인 |
| `hfrsda` | **DualAttn-TB** — 자체 설계 topology-blind 대조군. **HFRS-DA 구현이 아니다.** |

`hfrsda`라는 레거시 키 때문에 EXP-B의 `hfrsda` 결과가 논문 Table B에 "NutriGraphNet" 열로
실린 사고가 있었다(v1.3에서 수정). 두 variant는 별개다 — **`full`은 HealthGain@10 ≈ −0.010을
보고하고 `hfrsda`는 `None`을 보고하므로 이것으로 즉시 구분할 수 있다.**
귀속 고지는 논문 §4.1과 `src/nutrigraphnet_v2.py`의 `HFRSDAModel` docstring 참조.

---

## 알려진 측정 함정 (논문 §11)

- **HealthGain@K는 100% 밀도에서만 유효하다.** 부분추출이 `healthness` 엣지도 잘라내면
  엣지 없는 food가 0점이 되어 기준선이 0.6653 → 0.1529로 붕괴하고, 10%에서 **+0.503**이라는
  그럴듯하지만 완전히 거짓인 값이 나온다("희소할수록 건강해진다"로 오독됨).
- **재현성 바닥 ~0.005.** 동일 seed·설정에서도 GPU scatter 비결정성으로 ΔHR@10≈0.002,
  ΔAUC≈0.005가 발생한다. 이보다 작은 효과는 주장하지 않는다.
- **아키텍처 선택은 밀도를 넘어 전이되지 않는다.** 전밀도 벤치마크로 디코더를 고르면
  희소 영역에서 −12.6%인 설계를 채택하게 된다(§8.4).

---

## 저자

**Heejeong** — [@HeejeongH](https://github.com/HeejeongH)

## 라이선스

MIT License
