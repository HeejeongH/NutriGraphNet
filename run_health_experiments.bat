@echo off
REM Health-aware GNN 비교 실험 자동 실행 스크립트 (Windows)
chcp 65001 > nul
echo ============================================================
echo Health-aware GNN Comparative Experiments
echo ============================================================
echo.

REM 결과 저장 디렉토리 생성
if not exist results\health_experiments mkdir results\health_experiments

REM Experiment 1: Vanilla GNN (Baseline)
echo.
echo ============================================================
echo [1/6] Vanilla GNN (Baseline)
echo ============================================================
echo.

python train_v2.py --data_path data/processed_data/processed_data_GNN_fixed.pkl --model vanilla --hidden_channels 128 --out_channels 64 --epochs 50 --lr 0.001 --weight_decay 0.02 --loss standard --result_file results/health_experiments/vanilla_gnn_baseline.json

if %ERRORLEVEL% EQU 0 (
    echo [OK] Vanilla GNN ^(Baseline^) completed successfully
) else (
    echo [FAIL] Vanilla GNN ^(Baseline^) failed
)

echo.

REM Experiment 2: GraphSAGE (Baseline)
echo.
echo ============================================================
echo [2/6] GraphSAGE (Baseline)
echo ============================================================
echo.

python train_v2.py --data_path data/processed_data/processed_data_GNN_fixed.pkl --model graphsage --hidden_channels 128 --out_channels 64 --epochs 50 --lr 0.001 --weight_decay 0.02 --loss standard --result_file results/health_experiments/graphsage_baseline.json

if %ERRORLEVEL% EQU 0 (
    echo [OK] GraphSAGE ^(Baseline^) completed successfully
) else (
    echo [FAIL] GraphSAGE ^(Baseline^) failed
)

echo.

REM Experiment 3: GraphSAGE + Health Loss
echo.
echo ============================================================
echo [3/6] GraphSAGE + Health Loss
echo ============================================================
echo.

python train_v2.py --data_path data/processed_data/processed_data_GNN_fixed.pkl --model graphsage --hidden_channels 128 --out_channels 64 --epochs 50 --lr 0.001 --weight_decay 0.02 --loss health --health_lambda 0.1 --result_file results/health_experiments/graphsage_+_health_loss.json

if %ERRORLEVEL% EQU 0 (
    echo [OK] GraphSAGE + Health Loss completed successfully
) else (
    echo [FAIL] GraphSAGE + Health Loss failed
)

echo.

REM Experiment 4: NutriGraphNet V2 (Full)
echo.
echo ============================================================
echo [4/6] NutriGraphNet V2 (Full)
echo ============================================================
echo.

python train_v2.py --data_path data/processed_data/processed_data_GNN_fixed.pkl --model nutrigraphnet_v2 --hidden_channels 128 --out_channels 64 --epochs 50 --lr 0.001 --weight_decay 0.02 --loss adaptive --lambda_health_init 0.01 --lambda_health_max 0.1 --result_file results/health_experiments/nutrigraphnet_v2_full.json

if %ERRORLEVEL% EQU 0 (
    echo [OK] NutriGraphNet V2 ^(Full^) completed successfully
) else (
    echo [FAIL] NutriGraphNet V2 ^(Full^) failed
)

echo.

REM Experiment 5: NutriGraphNet V2 - Health Attention Only
echo.
echo ============================================================
echo [5/6] NutriGraphNet V2 - Health Attention Only
echo ============================================================
echo.

python train_v2.py --data_path data/processed_data/processed_data_GNN_fixed.pkl --model nutrigraphnet_v2 --hidden_channels 128 --out_channels 64 --epochs 50 --lr 0.001 --weight_decay 0.02 --loss standard --result_file results/health_experiments/nutrigraphnet_v2_-_health_attention_only.json

if %ERRORLEVEL% EQU 0 (
    echo [OK] NutriGraphNet V2 - Health Attention Only completed successfully
) else (
    echo [FAIL] NutriGraphNet V2 - Health Attention Only failed
)

echo.

REM Experiment 6: NutriGraphNet V2 - Health Loss Only
echo.
echo ============================================================
echo [6/6] NutriGraphNet V2 - Health Loss Only
echo ============================================================
echo.

python train_v2.py --data_path data/processed_data/processed_data_GNN_fixed.pkl --model nutrigraphnet_v2 --hidden_channels 128 --out_channels 64 --epochs 50 --lr 0.001 --weight_decay 0.02 --loss adaptive --lambda_health_init 0.01 --lambda_health_max 0.1 --result_file results/health_experiments/nutrigraphnet_v2_-_health_loss_only.json

if %ERRORLEVEL% EQU 0 (
    echo [OK] NutriGraphNet V2 - Health Loss Only completed successfully
) else (
    echo [FAIL] NutriGraphNet V2 - Health Loss Only failed
)

echo.

echo ============================================================
echo All experiments completed!
echo ============================================================
echo.

REM 결과 비교
echo Generating comparison report...
python src\compare_health_results.py

echo.
echo Done!
pause
