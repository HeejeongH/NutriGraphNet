@echo off
:: NOTE (2026-07-15): 2026-07-13에 GPU 머신에서 실행 완료된 스크립트 (provenance 보존용).
::  - EXP-C 섹션 산출물 = results/gpu/C_lambda_* → 논문 Table C-Full (EXP-C-Full)의 원천 데이터
::  - EXP-G 섹션 산출물은 num_layers 미반영 문제로 격리됨 → results/gpu/_stale/README_WHY_STALE.md 참고
::    (논문 Table G는 results/analysis/G_layers_* 기반이라 영향 없음)
echo ============================================================
echo  NutriGraphNet GPU Experiments - Full Pipeline
echo ============================================================
echo  Started: %date% %time%
echo ============================================================

:: ===== EXP-C: lambda_health sweep =====
echo.
echo [1/3] EXP-C: lambda_health sweep (NutriGraphNet)

python src\nutrigraphnet_v2.py --variants full --lambda_health 0.0   --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/C_lambda_0.0
echo   Done lambda=0.0

python src\nutrigraphnet_v2.py --variants full --lambda_health 0.001 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/C_lambda_0.001
echo   Done lambda=0.001

python src\nutrigraphnet_v2.py --variants full --lambda_health 0.005 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/C_lambda_0.005
echo   Done lambda=0.005

python src\nutrigraphnet_v2.py --variants full --lambda_health 0.01  --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/C_lambda_0.01
echo   Done lambda=0.01

python src\nutrigraphnet_v2.py --variants full --lambda_health 0.05  --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/C_lambda_0.05
echo   Done lambda=0.05

python src\nutrigraphnet_v2.py --variants full --lambda_health 0.1   --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/C_lambda_0.1
echo   Done lambda=0.1

python src\nutrigraphnet_v2.py --variants full --lambda_health 0.5   --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/C_lambda_0.5
echo   Done lambda=0.5

python src\nutrigraphnet_v2.py --variants full --lambda_health 1.0   --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/C_lambda_1.0
echo   Done lambda=1.0

echo [EXP-C] Complete!

:: ===== EXP-B: Sparsity sweep =====
echo.
echo [2/3] EXP-B: Sparsity sweep (mf, lightgcn, ngcf, sgl, hfrsda)

python src\nutrigraphnet_v2.py --variants mf,lightgcn,ngcf,sgl,hfrsda --interaction_ratio 0.1 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/B_sparsity_10pct
echo   Done sparsity=10%%

python src\nutrigraphnet_v2.py --variants mf,lightgcn,ngcf,sgl,hfrsda --interaction_ratio 0.3 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/B_sparsity_30pct
echo   Done sparsity=30%%

python src\nutrigraphnet_v2.py --variants mf,lightgcn,ngcf,sgl,hfrsda --interaction_ratio 0.5 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/B_sparsity_50pct
echo   Done sparsity=50%%

python src\nutrigraphnet_v2.py --variants mf,lightgcn,ngcf,sgl,hfrsda --interaction_ratio 0.7 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/B_sparsity_70pct
echo   Done sparsity=70%%

python src\nutrigraphnet_v2.py --variants mf,lightgcn,ngcf,sgl,hfrsda --interaction_ratio 1.0 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --num_layers 3 --heads 4 --seed 42 --output_dir results/gpu/B_sparsity_100pct
echo   Done sparsity=100%%

echo [EXP-B] Complete!

:: ===== EXP-G: Layer depth sweep =====
echo.
echo [3/3] EXP-G: Layer depth sweep (lightgcn, ngcf)

python src\nutrigraphnet_v2.py --variants lightgcn,ngcf --num_layers 1 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --heads 4 --seed 42 --output_dir results/gpu/G_layers_1
echo   Done num_layers=1

python src\nutrigraphnet_v2.py --variants lightgcn,ngcf --num_layers 2 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --heads 4 --seed 42 --output_dir results/gpu/G_layers_2
echo   Done num_layers=2

python src\nutrigraphnet_v2.py --variants lightgcn,ngcf --num_layers 3 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --heads 4 --seed 42 --output_dir results/gpu/G_layers_3
echo   Done num_layers=3

python src\nutrigraphnet_v2.py --variants lightgcn,ngcf --num_layers 4 --n_folds 5 --epochs 300 --patience 30 --hidden_channels 128 --out_channels 64 --heads 4 --seed 42 --output_dir results/gpu/G_layers_4
echo   Done num_layers=4

echo [EXP-G] Complete!

:: ===== 결과 요약 =====
echo.
echo ============================================================
echo  All experiments complete!
echo  Finished: %date% %time%
echo ============================================================
python src\run_analysis_experiments.py --exp summary
echo.
echo Results saved to: results/gpu/
echo Upload SUMMARY.json to update the paper!
pause
