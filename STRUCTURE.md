# 📁 NutriGraphNet 프로젝트 구조

```
NutriGraphNet/
├── 📄 train_v2.py                    # 메인 학습 스크립트
├── 🔧 run_health_experiments.sh      # 실험 자동화 스크립트
├── 📖 README.md                      # 프로젝트 문서
├── 📋 requirements.txt               # 패키지 의존성
├── 🔒 .gitignore                     # Git 무시 파일
│
├── 📁 src/                           # 핵심 소스 코드
│   ├── 🧠 NutriGraphNet_v2.py        # 최신 GNN 모델 (V2)
│   ├── 💚 HealthAwareGNN.py          # Health-aware GNN 모델
│   ├── 🏥 health_score_calculator.py # 개인별 건강 점수 계산
│   ├── 📊 evaluation_metrics.py      # Health-aware 평가 메트릭
│   ├── 🛠️ training_utils.py          # 학습 유틸리티
│   ├── 🔬 run_health_aware_experiments.py  # 실험 생성 스크립트
│   ├── 📈 compare_health_results.py  # 결과 비교 및 시각화
│   └── ✅ check_pipeline.py          # 파이프라인 검증
│
├── 📁 data/
│   ├── 🔧 graph_builder.py           # 그래프 데이터 생성 스크립트
│   └── 📁 processed_data/
│       └── 💾 processed_data_GNN_fixed.pkl  # 정규화된 최신 데이터 (39MB)
│
└── 📁 etc/
    ├── 📁 old_data_scripts/          # 구버전 데이터 스크립트 백업
    │   ├── build_graph_data.py
    │   └── graph_builder.py
    └── 📁 old_versions/               # 구버전 파일 아카이브 (Git 무시)
        ├── NutriGraphNet.py           # V1 모델
        ├── train_v2.py                # 중복 파일
        ├── health_aware_gnn.ipynb     # Jupyter notebook
        ├── fold_1.pkl ~ fold_5.pkl    # K-fold 데이터 (450MB)
        ├── processed_data_GNN.pkl     # 구버전 데이터 (37MB)
        └── processed_data_GNN_cpu.pkl # 구버전 데이터 (37MB)
```

## 📊 파일 크기 정리

**활성 파일 (필수):**
- `processed_data_GNN_fixed.pkl`: 39MB (정규화된 최신 데이터)
- 소스 코드: 약 112KB

**아카이브 파일 (etc/old_versions/):**
- 구버전 데이터: 524MB
- Git에서 제외됨 (.gitignore)

**총 디스크 절약: 524MB → Git 리포지토리 크기 감소**

## 🎯 핵심 워크플로우

1. **파이프라인 검증**: `python src/check_pipeline.py`
2. **실험 생성**: `python src/run_health_aware_experiments.py --epochs 50`
3. **실험 실행**: `bash run_health_experiments.sh`
4. **결과 비교**: `python src/compare_health_results.py`

## 📝 참고

- 모든 소스 코드는 `src/` 디렉토리에 통합
- 중복/구버전 파일은 `etc/old_versions/`로 이동
- Git 리포지토리는 필수 파일만 포함
- 로컬 환경에서는 필요시 old_versions 파일 참조 가능
