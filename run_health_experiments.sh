#!/bin/bash
# Health-aware GNN 비교 실험 자동 실행 스크립트
# 생성 일자: $(date)

set -e  # 에러 발생 시 중단

echo '='
echo '🧪 Health-aware GNN Comparative Experiments'
echo '='
echo ''

# 결과 저장 디렉토리
mkdir -p results/health_experiments

# Experiment 1: Vanilla GNN (Baseline)
echo ''
echo '============================================================'
echo '📊 [1/6] Vanilla GNN (Baseline)'
echo '============================================================'
echo ''

python train_v2.py --data_path data/processed_data/processed_data_GNN_fixed.pkl --model vanilla --hidden_channels 128 --out_channels 64 --epochs 30 --lr 0.001 --weight_decay 0.02 --loss standard --result_file results/health_experiments/vanilla_gnn_baseline.json

if [ $? -eq 0 ]; then
    echo '✅ Vanilla GNN (Baseline) completed successfully'
else
    echo '❌ Vanilla GNN (Baseline) failed'
fi

echo ''

# Experiment 2: GraphSAGE (Baseline)
echo ''
echo '============================================================'
echo '📊 [2/6] GraphSAGE (Baseline)'
echo '============================================================'
echo ''

python train_v2.py --data_path data/processed_data/processed_data_GNN_fixed.pkl --model graphsage --hidden_channels 128 --out_channels 64 --epochs 30 --lr 0.001 --weight_decay 0.02 --loss standard --result_file results/health_experiments/graphsage_baseline.json

if [ $? -eq 0 ]; then
    echo '✅ GraphSAGE (Baseline) completed successfully'
else
    echo '❌ GraphSAGE (Baseline) failed'
fi

echo ''

# Experiment 3: GraphSAGE + Health Loss
echo ''
echo '============================================================'
echo '📊 [3/6] GraphSAGE + Health Loss'
echo '============================================================'
echo ''

python train_v2.py --data_path data/processed_data/processed_data_GNN_fixed.pkl --model graphsage --hidden_channels 128 --out_channels 64 --epochs 30 --lr 0.001 --weight_decay 0.02 --loss health --health_lambda 0.1 --result_file results/health_experiments/graphsage_+_health_loss.json

if [ $? -eq 0 ]; then
    echo '✅ GraphSAGE + Health Loss completed successfully'
else
    echo '❌ GraphSAGE + Health Loss failed'
fi

echo ''

# Experiment 4: NutriGraphNet V2 (Full)
echo ''
echo '============================================================'
echo '📊 [4/6] NutriGraphNet V2 (Full)'
echo '============================================================'
echo ''

python train_v2.py --data_path data/processed_data/processed_data_GNN_fixed.pkl --model nutrigraphnet_v2 --hidden_channels 128 --out_channels 64 --epochs 30 --lr 0.001 --weight_decay 0.02 --loss adaptive --lambda_health_init 0.01 --lambda_health_max 0.1 --result_file results/health_experiments/nutrigraphnet_v2_full.json

if [ $? -eq 0 ]; then
    echo '✅ NutriGraphNet V2 (Full) completed successfully'
else
    echo '❌ NutriGraphNet V2 (Full) failed'
fi

echo ''

# Experiment 5: NutriGraphNet V2 - Health Attention Only
echo ''
echo '============================================================'
echo '📊 [5/6] NutriGraphNet V2 - Health Attention Only'
echo '============================================================'
echo ''

python train_v2.py --data_path data/processed_data/processed_data_GNN_fixed.pkl --model nutrigraphnet_v2 --hidden_channels 128 --out_channels 64 --epochs 30 --lr 0.001 --weight_decay 0.02 --loss standard --result_file results/health_experiments/nutrigraphnet_v2_-_health_attention_only.json

if [ $? -eq 0 ]; then
    echo '✅ NutriGraphNet V2 - Health Attention Only completed successfully'
else
    echo '❌ NutriGraphNet V2 - Health Attention Only failed'
fi

echo ''

# Experiment 6: NutriGraphNet V2 - Health Loss Only
echo ''
echo '============================================================'
echo '📊 [6/6] NutriGraphNet V2 - Health Loss Only'
echo '============================================================'
echo ''

python train_v2.py --data_path data/processed_data/processed_data_GNN_fixed.pkl --model nutrigraphnet_v2 --hidden_channels 128 --out_channels 64 --epochs 30 --lr 0.001 --weight_decay 0.02 --loss adaptive --lambda_health_init 0.01 --lambda_health_max 0.1 --result_file results/health_experiments/nutrigraphnet_v2_-_health_loss_only.json

if [ $? -eq 0 ]; then
    echo '✅ NutriGraphNet V2 - Health Loss Only completed successfully'
else
    echo '❌ NutriGraphNet V2 - Health Loss Only failed'
fi

echo ''

echo '='
echo '✅ All experiments completed!'
echo '='
echo ''

# 결과 비교
echo 'Generating comparison report...'
python compare_health_results.py
