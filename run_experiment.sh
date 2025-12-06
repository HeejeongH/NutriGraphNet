#!/bin/bash

# NutriGraphNet V2 실험 실행 스크립트

echo "🚀 NutriGraphNet V2 Training Script"
echo "===================================="
echo ""

# 기본 설정
DATA_PATH="../data/processed_data/processed_data_GNN.pkl"
SAVE_DIR="./results"

# 결과 디렉토리 생성
mkdir -p $SAVE_DIR

# 1. Quick Test (빠른 테스트 - 작은 모델)
echo "📝 Experiment 1: Quick Test (Small Model)"
python train_v2.py \
    --data_path $DATA_PATH \
    --hidden_channels 128 \
    --out_channels 64 \
    --num_layers 2 \
    --epochs 30 \
    --lr 0.001 \
    --lambda_health_init 0.01 \
    --lambda_health_max 0.05 \
    --save_path $SAVE_DIR/quick_test.pth \
    --print_every 5

echo ""
echo "✅ Quick test completed!"
echo ""

# 2. V2 Full Model (전체 V2 기능)
echo "📝 Experiment 2: V2 Full Model"
python train_v2.py \
    --data_path $DATA_PATH \
    --hidden_channels 256 \
    --out_channels 128 \
    --num_layers 3 \
    --epochs 100 \
    --lr 0.001 \
    --lambda_health_init 0.01 \
    --lambda_health_max 0.1 \
    --focal_gamma 2.0 \
    --temperature 2.0 \
    --patience 15 \
    --save_path $SAVE_DIR/v2_full.pth \
    --print_every 5

echo ""
echo "✅ V2 Full model training completed!"
echo ""

# 3. V2 with Lower Health Weight (건강 가중치 낮춤)
echo "📝 Experiment 3: V2 with Lower Health Weight"
python train_v2.py \
    --data_path $DATA_PATH \
    --hidden_channels 256 \
    --out_channels 128 \
    --num_layers 3 \
    --epochs 100 \
    --lr 0.001 \
    --lambda_health_init 0.005 \
    --lambda_health_max 0.05 \
    --focal_gamma 2.0 \
    --temperature 2.0 \
    --save_path $SAVE_DIR/v2_low_health.pth \
    --print_every 5

echo ""
echo "✅ Low health weight experiment completed!"
echo ""

# 4. V2 with Higher Capacity (더 큰 모델)
echo "📝 Experiment 4: V2 with Higher Capacity"
python train_v2.py \
    --data_path $DATA_PATH \
    --hidden_channels 512 \
    --out_channels 256 \
    --num_layers 4 \
    --epochs 100 \
    --lr 0.0005 \
    --lambda_health_init 0.01 \
    --lambda_health_max 0.1 \
    --dropout 0.4 \
    --save_path $SAVE_DIR/v2_large.pth \
    --print_every 5

echo ""
echo "✅ Large model experiment completed!"
echo ""

echo "🎉 All experiments completed!"
echo "Results saved in: $SAVE_DIR"
echo ""
echo "📊 To compare results, check the .pth files in results/"
