"""
Health-aware 모델 비교 실험 스크립트

연구 목적:
1. Baseline 모델 (선호도만)
2. Health-aware 모델 (선호도 + 건강도)
3. 두 모델의 성능 비교 (선호도 예측 + 건강도 고려)
"""

import torch
import pickle
import argparse
import json
from pathlib import Path
import sys

sys.path.append('src')

from evaluation_metrics import (
    compute_comprehensive_metrics,
    print_metrics_comparison,
    compute_health_aware_ranking_metrics
)

# 기본 설정
DEFAULT_CONFIG = {
    'data_path': 'data/processed_data/processed_data_GNN_fixed.pkl',
    'hidden_channels': 128,
    'out_channels': 64,
    'epochs': 50,
    'lr': 0.001,
    'weight_decay': 0.02,
    'test_ratio': 0.2,
    'seed': 42
}


def run_single_experiment(model_type, use_health_attention, use_health_loss, config):
    """
    단일 실험 실행
    
    Args:
        model_type: 'vanilla', 'graphsage', 'gat', 'nutrigraphnet_v2'
        use_health_attention: Health attention 사용 여부
        use_health_loss: Health loss 사용 여부
        config: 설정 dict
        
    Returns:
        dict: 실험 결과
    """
    
    print(f"\n{'='*80}")
    print(f"🚀 Running Experiment:")
    print(f"   Model: {model_type}")
    print(f"   Health Attention: {'✅' if use_health_attention else '❌'}")
    print(f"   Health Loss: {'✅' if use_health_loss else '❌'}")
    print(f"{'='*80}\n")
    
    # 실제 학습은 train_v2.py에서 실행
    # 여기서는 실험 설정만 반환
    
    experiment_config = {
        'model_type': model_type,
        'use_health_attention': use_health_attention,
        'use_health_loss': use_health_loss,
        'config': config
    }
    
    # train_v2.py 호출을 위한 명령어 생성
    cmd_parts = [
        'python train_v2.py',
        f'--data_path {config["data_path"]}',
        f'--model {model_type}',
        f'--hidden_channels {config["hidden_channels"]}',
        f'--out_channels {config["out_channels"]}',
        f'--epochs {config["epochs"]}',
        f'--lr {config["lr"]}',
        f'--weight_decay {config["weight_decay"]}',
    ]
    
    # Health loss 설정
    if use_health_loss:
        if model_type == 'nutrigraphnet_v2':
            cmd_parts.append('--loss adaptive')
            cmd_parts.append('--lambda_health_init 0.01')
            cmd_parts.append('--lambda_health_max 0.1')
        else:
            cmd_parts.append('--loss health')
            cmd_parts.append('--health_lambda 0.1')
    else:
        cmd_parts.append('--loss standard')
    
    command = ' '.join(cmd_parts)
    
    return {
        'experiment_config': experiment_config,
        'command': command
    }


def run_comparative_experiments(config):
    """
    비교 실험 실행
    
    실험 세트:
    1. Baseline (GraphSAGE, no health)
    2. Health-aware (GraphSAGE + health attention + health loss)
    3. NutriGraphNet V2 (full model)
    """
    
    experiments = []
    
    # ============================================
    # Experiment 1: Baseline - GraphSAGE (선호도만)
    # ============================================
    print("\n" + "="*80)
    print("📋 Experiment Set 1: Baseline Models (Preference Only)")
    print("="*80)
    
    baseline_experiments = [
        ('vanilla', False, False, 'Vanilla GNN (Baseline)'),
        ('graphsage', False, False, 'GraphSAGE (Baseline)'),
    ]
    
    for model, use_attn, use_loss, name in baseline_experiments:
        exp = run_single_experiment(model, use_attn, use_loss, config)
        exp['experiment_name'] = name
        exp['category'] = 'baseline'
        experiments.append(exp)
    
    # ============================================
    # Experiment 2: Health-aware Models
    # ============================================
    print("\n" + "="*80)
    print("📋 Experiment Set 2: Health-aware Models")
    print("="*80)
    
    health_experiments = [
        ('graphsage', False, True, 'GraphSAGE + Health Loss'),
        ('nutrigraphnet_v2', True, True, 'NutriGraphNet V2 (Full)'),
    ]
    
    for model, use_attn, use_loss, name in health_experiments:
        exp = run_single_experiment(model, use_attn, use_loss, config)
        exp['experiment_name'] = name
        exp['category'] = 'health_aware'
        experiments.append(exp)
    
    # ============================================
    # Ablation Studies (선택)
    # ============================================
    print("\n" + "="*80)
    print("📋 Experiment Set 3: Ablation Studies (Optional)")
    print("="*80)
    
    ablation_experiments = [
        ('nutrigraphnet_v2', True, False, 'NutriGraphNet V2 - Health Attention Only'),
        ('nutrigraphnet_v2', False, True, 'NutriGraphNet V2 - Health Loss Only'),
    ]
    
    for model, use_attn, use_loss, name in ablation_experiments:
        exp = run_single_experiment(model, use_attn, use_loss, config)
        exp['experiment_name'] = name
        exp['category'] = 'ablation'
        experiments.append(exp)
    
    return experiments


def generate_experiment_script(experiments, output_file='run_health_experiments.sh'):
    """
    실험 실행 스크립트 생성
    """
    
    script_lines = [
        "#!/bin/bash",
        "# Health-aware GNN 비교 실험 자동 실행 스크립트",
        "# 생성 일자: $(date)",
        "",
        "set -e  # 에러 발생 시 중단",
        "",
        "echo '='",
        "echo '🧪 Health-aware GNN Comparative Experiments'",
        "echo '='",
        "echo ''",
        "",
        "# 결과 저장 디렉토리",
        "mkdir -p results/health_experiments",
        ""
    ]
    
    for i, exp in enumerate(experiments, 1):
        name = exp['experiment_name']
        cmd = exp['command']
        category = exp['category']
        
        # 결과 파일명
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        result_file = f"results/health_experiments/{safe_name}.json"
        
        script_lines.extend([
            f"# Experiment {i}: {name}",
            f"echo ''",
            f"echo '{'='*60}'",
            f"echo '📊 [{i}/{len(experiments)}] {name}'",
            f"echo '{'='*60}'",
            f"echo ''",
            "",
            f"{cmd} --result_file {result_file}",
            "",
            f"if [ $? -eq 0 ]; then",
            f"    echo '✅ {name} completed successfully'",
            f"else",
            f"    echo '❌ {name} failed'",
            f"fi",
            "",
            "echo ''",
            ""
        ])
    
    script_lines.extend([
        "echo '='",
        "echo '✅ All experiments completed!'",
        "echo '='",
        "echo ''",
        "",
        "# 결과 비교",
        "echo 'Generating comparison report...'",
        "python compare_health_results.py",
        ""
    ])
    
    # 스크립트 저장
    with open(output_file, 'w') as f:
        f.write('\n'.join(script_lines))
    
    # 실행 권한 부여
    import os
    os.chmod(output_file, 0o755)
    
    print(f"\n✅ Experiment script generated: {output_file}")
    print(f"   Total experiments: {len(experiments)}")
    print(f"\n🚀 To run all experiments:")
    print(f"   bash {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Health-aware GNN Comparative Experiments')
    
    parser.add_argument('--data_path', type=str, 
                       default=DEFAULT_CONFIG['data_path'],
                       help='Path to processed data')
    parser.add_argument('--hidden_channels', type=int, 
                       default=DEFAULT_CONFIG['hidden_channels'],
                       help='Hidden channels')
    parser.add_argument('--out_channels', type=int, 
                       default=DEFAULT_CONFIG['out_channels'],
                       help='Output channels')
    parser.add_argument('--epochs', type=int, 
                       default=DEFAULT_CONFIG['epochs'],
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, 
                       default=DEFAULT_CONFIG['lr'],
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, 
                       default=DEFAULT_CONFIG['weight_decay'],
                       help='Weight decay')
    parser.add_argument('--output_script', type=str, 
                       default='run_health_experiments.sh',
                       help='Output shell script name')
    
    args = parser.parse_args()
    
    # Config 구성
    config = {
        'data_path': args.data_path,
        'hidden_channels': args.hidden_channels,
        'out_channels': args.out_channels,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'test_ratio': DEFAULT_CONFIG['test_ratio'],
        'seed': DEFAULT_CONFIG['seed']
    }
    
    # 실험 생성
    experiments = run_comparative_experiments(config)
    
    # 스크립트 생성
    generate_experiment_script(experiments, args.output_script)
    
    # 실험 요약 저장
    summary = {
        'config': config,
        'experiments': [
            {
                'name': exp['experiment_name'],
                'category': exp['category'],
                'model_type': exp['experiment_config']['model_type'],
                'use_health_attention': exp['experiment_config']['use_health_attention'],
                'use_health_loss': exp['experiment_config']['use_health_loss']
            }
            for exp in experiments
        ]
    }
    
    # 디렉토리 생성
    summary_dir = Path('results/health_experiments')
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = summary_dir / 'experiment_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Experiment summary saved: results/health_experiments/experiment_summary.json")
    
    # 실험 목록 출력
    print(f"\n{'='*80}")
    print("📋 Experiment Summary")
    print(f"{'='*80}\n")
    
    categories = {}
    for exp in experiments:
        cat = exp['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(exp['experiment_name'])
    
    for cat, names in categories.items():
        print(f"\n{cat.upper()}:")
        for name in names:
            print(f"   • {name}")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
