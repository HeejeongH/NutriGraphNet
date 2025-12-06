#!/bin/bash
# Run all experiments for model comparison

echo "=================================="
echo "🚀 NutriGraphNet Batch Experiments"
echo "=================================="
echo ""

# Data path
DATA_PATH="data/processed_data/processed_data_GNN_cpu.pkl"
EPOCHS=50

# Create results directory
mkdir -p results/

echo "📊 Experiment 1/8: Vanilla GNN with Standard Loss"
python train_v2.py \
  --data_path $DATA_PATH \
  --model vanilla \
  --loss standard \
  --epochs $EPOCHS \
  --hidden_channels 128 \
  --out_channels 64 \
  --save_path results/vanilla_standard.pth

echo ""
echo "📊 Experiment 2/8: GraphSAGE with Standard Loss"
python train_v2.py \
  --data_path $DATA_PATH \
  --model graphsage \
  --loss standard \
  --epochs $EPOCHS \
  --hidden_channels 128 \
  --out_channels 64 \
  --save_path results/graphsage_standard.pth

echo ""
echo "📊 Experiment 3/8: GAT with Standard Loss"
python train_v2.py \
  --data_path $DATA_PATH \
  --model gat \
  --loss standard \
  --epochs $EPOCHS \
  --hidden_channels 128 \
  --out_channels 64 \
  --save_path results/gat_standard.pth

echo ""
echo "📊 Experiment 4/8: Vanilla GNN with Focal Loss"
python train_v2.py \
  --data_path $DATA_PATH \
  --model vanilla \
  --loss focal \
  --epochs $EPOCHS \
  --hidden_channels 128 \
  --out_channels 64 \
  --save_path results/vanilla_focal.pth

echo ""
echo "📊 Experiment 5/8: GAT with Health Loss"
python train_v2.py \
  --data_path $DATA_PATH \
  --model gat \
  --loss health \
  --health_lambda 0.1 \
  --epochs $EPOCHS \
  --hidden_channels 128 \
  --out_channels 64 \
  --save_path results/gat_health.pth

echo ""
echo "📊 Experiment 6/8: NutriGraphNet V2 with Adaptive Loss"
python train_v2.py \
  --data_path $DATA_PATH \
  --model nutrigraphnet_v2 \
  --loss adaptive \
  --epochs 100 \
  --hidden_channels 256 \
  --out_channels 128 \
  --lambda_health_init 0.01 \
  --lambda_health_max 0.1 \
  --save_path results/nutrigraphnet_v2_adaptive.pth

echo ""
echo "📊 Experiment 7/8: Health GNN with Health Loss"
python train_v2.py \
  --data_path $DATA_PATH \
  --model health_gnn \
  --loss health \
  --health_lambda 0.1 \
  --epochs 100 \
  --hidden_channels 256 \
  --out_channels 128 \
  --save_path results/health_gnn_health.pth

echo ""
echo "📊 Experiment 8/8: NutriGraphNet V2 (Full Configuration)"
python train_v2.py \
  --data_path $DATA_PATH \
  --model nutrigraphnet_v2 \
  --loss adaptive \
  --epochs 100 \
  --hidden_channels 256 \
  --out_channels 128 \
  --num_layers 3 \
  --lambda_health_init 0.01 \
  --lambda_health_max 0.1 \
  --focal_gamma 2.0 \
  --temperature 2.0 \
  --save_path results/nutrigraphnet_v2_full.pth

echo ""
echo "=================================="
echo "✅ All experiments completed!"
echo "=================================="
echo ""
echo "Results saved in results/ directory"
echo "Run 'python compare_results.py' to compare all experiments"
