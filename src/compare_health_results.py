"""
Health-aware 실험 결과 비교 및 시각화
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# 한글 폰트 설정 (Mac/Linux용)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_experiment_results(result_dir='results/health_experiments'):
    """실험 결과 로드"""
    
    result_path = Path(result_dir)
    
    if not result_path.exists():
        print(f"⚠️  Result directory not found: {result_dir}")
        return None
    
    results = []
    
    for json_file in result_path.glob('*.json'):
        if json_file.name == 'experiment_summary.json':
            continue
        
        with open(json_file, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    if len(results) == 0:
        print(f"⚠️  No result files found in {result_dir}")
        return None
    
    return results


def create_comparison_dataframe(results):
    """결과를 DataFrame으로 변환"""
    
    rows = []
    
    for result in results:
        # 모델 정보
        model_info = result.get('model_info', {})
        model_type = model_info.get('type', 'unknown')
        has_health_attention = model_info.get('health_attention', False)
        has_health_loss = model_info.get('health_loss', False)
        
        # 메트릭
        metrics = result.get('test_metrics', {})
        
        # Health-aware 여부 판단
        is_health_aware = has_health_attention or has_health_loss
        
        row = {
            'Model': model_type.upper(),
            'Health_Aware': 'Yes' if is_health_aware else 'No',
            'Health_Attention': has_health_attention,
            'Health_Loss': has_health_loss,
            **metrics
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def plot_comparison(df, output_dir='results/health_experiments'):
    """비교 시각화"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 선호도 예측 메트릭 비교
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Preference Prediction Metrics Comparison', fontsize=16, fontweight='bold')
    
    preference_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    for i, metric in enumerate(preference_metrics):
        ax = axes[i // 3, i % 3]
        
        if metric in df.columns:
            sns.barplot(data=df, x='Model', y=metric, hue='Health_Aware', ax=ax)
            ax.set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_ylabel(metric.upper())
            ax.grid(axis='y', alpha=0.3)
            
            # 값 표시
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', padding=3)
    
    # 마지막 subplot 제거
    fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(output_path / 'preference_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path / 'preference_metrics_comparison.png'}")
    plt.close()
    
    # 2. 건강도 메트릭 비교
    health_metrics = ['avg_health_score', 'health_precision', 'health_aware_recall', 
                      'health_improvement', 'health_aware_f1']
    
    available_health_metrics = [m for m in health_metrics if m in df.columns]
    
    if len(available_health_metrics) > 0:
        n_metrics = len(available_health_metrics)
        n_rows = (n_metrics + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6*n_rows))
        fig.suptitle('Health-awareness Metrics Comparison', fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, metric in enumerate(available_health_metrics):
            ax = axes[i // 3, i % 3]
            
            sns.barplot(data=df, x='Model', y=metric, hue='Health_Aware', ax=ax)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.grid(axis='y', alpha=0.3)
            
            # 값 표시
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', padding=3)
        
        # 빈 subplot 제거
        for i in range(len(available_health_metrics), n_rows * 3):
            fig.delaxes(axes[i // 3, i % 3])
        
        plt.tight_layout()
        plt.savefig(output_path / 'health_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path / 'health_metrics_comparison.png'}")
        plt.close()
    
    # 3. Radar Chart (종합 비교)
    create_radar_chart(df, output_path)
    
    # 4. Top-K 메트릭 비교
    plot_topk_metrics(df, output_path)


def create_radar_chart(df, output_path):
    """레이더 차트 생성"""
    
    # 주요 메트릭 선택
    key_metrics = ['f1', 'auc', 'avg_health_score', 'health_precision', 'health_aware_f1']
    available_metrics = [m for m in key_metrics if m in df.columns]
    
    if len(available_metrics) < 3:
        print("⚠️  Not enough metrics for radar chart")
        return
    
    # 각 모델별 평균값
    baseline_df = df[df['Health_Aware'] == 'No']
    health_df = df[df['Health_Aware'] == 'Yes']
    
    if len(baseline_df) == 0 or len(health_df) == 0:
        print("⚠️  Missing baseline or health-aware models for radar chart")
        return
    
    baseline_values = [baseline_df[m].mean() for m in available_metrics]
    health_values = [health_df[m].mean() for m in available_metrics]
    
    # Radar chart
    labels = [m.replace('_', ' ').title() for m in available_metrics]
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    baseline_values += baseline_values[:1]
    health_values += health_values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, baseline_values, 'o-', linewidth=2, label='Baseline', color='skyblue')
    ax.fill(angles, baseline_values, alpha=0.25, color='skyblue')
    
    ax.plot(angles, health_values, 'o-', linewidth=2, label='Health-aware', color='salmon')
    ax.fill(angles, health_values, alpha=0.25, color='salmon')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_title('Baseline vs Health-aware Models', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path / 'radar_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path / 'radar_comparison.png'}")
    plt.close()


def plot_topk_metrics(df, output_path):
    """Top-K 메트릭 시각화"""
    
    topk_columns = [col for col in df.columns if col.startswith(('accuracy@', 'health@', 'ndcg@'))]
    
    if len(topk_columns) == 0:
        print("⚠️  No Top-K metrics found")
        return
    
    # K 값별로 그룹화
    k_values = sorted(list(set([int(col.split('@')[1]) for col in topk_columns])))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Top-K Metrics Comparison', fontsize=16, fontweight='bold')
    
    metric_types = ['accuracy', 'health', 'ndcg']
    
    for i, metric_type in enumerate(metric_types):
        ax = axes[i]
        
        for model_name in df['Model'].unique():
            model_df = df[df['Model'] == model_name]
            
            values = []
            for k in k_values:
                col = f'{metric_type}@{k}'
                if col in df.columns:
                    values.append(model_df[col].values[0])
                else:
                    values.append(0)
            
            linestyle = '--' if model_df['Health_Aware'].values[0] == 'No' else '-'
            marker = 'o' if model_df['Health_Aware'].values[0] == 'No' else 's'
            
            ax.plot(k_values, values, marker=marker, linestyle=linestyle, 
                   linewidth=2, markersize=8, label=model_name)
        
        ax.set_xlabel('K', fontsize=12)
        ax.set_ylabel(f'{metric_type.title()}@K', fontsize=12)
        ax.set_title(f'{metric_type.title()}@K Comparison', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'topk_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path / 'topk_metrics_comparison.png'}")
    plt.close()


def print_text_report(df, output_path):
    """텍스트 리포트 생성"""
    
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("HEALTH-AWARE GNN EXPERIMENT RESULTS")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 1. Baseline vs Health-aware 비교
    baseline_df = df[df['Health_Aware'] == 'No']
    health_df = df[df['Health_Aware'] == 'Yes']
    
    if len(baseline_df) > 0 and len(health_df) > 0:
        report_lines.append("1. BASELINE VS HEALTH-AWARE COMPARISON")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        key_metrics = ['f1', 'auc', 'avg_health_score', 'health_precision', 'health_aware_f1']
        available_metrics = [m for m in key_metrics if m in df.columns]
        
        report_lines.append(f"{'Metric':<25} {'Baseline':>15} {'Health-aware':>15} {'Improvement':>15}")
        report_lines.append("-" * 80)
        
        for metric in available_metrics:
            baseline_val = baseline_df[metric].mean()
            health_val = health_df[metric].mean()
            improvement = ((health_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
            
            report_lines.append(
                f"{metric:<25} {baseline_val:>15.4f} {health_val:>15.4f} {improvement:>14.2f}%"
            )
        
        report_lines.append("")
    
    # 2. 개별 모델 결과
    report_lines.append("2. INDIVIDUAL MODEL RESULTS")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    for _, row in df.iterrows():
        report_lines.append(f"Model: {row['Model']}")
        report_lines.append(f"  Health-aware: {row['Health_Aware']}")
        report_lines.append(f"  Preference Metrics:")
        report_lines.append(f"    F1 Score:  {row.get('f1', 0):.4f}")
        report_lines.append(f"    AUC:       {row.get('auc', 0):.4f}")
        report_lines.append(f"  Health Metrics:")
        report_lines.append(f"    Avg Health Score:    {row.get('avg_health_score', 0):.4f}")
        report_lines.append(f"    Health Precision:    {row.get('health_precision', 0):.4f}")
        report_lines.append(f"    Health-aware F1:     {row.get('health_aware_f1', 0):.4f}")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    
    # 파일 저장
    report_text = '\n'.join(report_lines)
    
    with open(output_path / 'experiment_report.txt', 'w') as f:
        f.write(report_text)
    
    print(f"\n✅ Saved: {output_path / 'experiment_report.txt'}")
    
    # 콘솔 출력
    print(f"\n{report_text}")


def main():
    print("\n" + "="*80)
    print("📊 Comparing Health-aware Experiment Results")
    print("="*80 + "\n")
    
    # 결과 로드
    results = load_experiment_results()
    
    if results is None:
        print("❌ No results to compare. Run experiments first!")
        return
    
    print(f"✅ Loaded {len(results)} experiment results\n")
    
    # DataFrame 생성
    df = create_comparison_dataframe(results)
    
    # CSV 저장
    output_path = Path('results/health_experiments')
    df.to_csv(output_path / 'comparison_results.csv', index=False)
    print(f"✅ Saved: {output_path / 'comparison_results.csv'}")
    
    # 시각화
    print("\n📈 Generating visualizations...")
    plot_comparison(df, output_dir='results/health_experiments')
    
    # 텍스트 리포트
    print("\n📝 Generating text report...")
    print_text_report(df, output_path)
    
    print("\n" + "="*80)
    print("✅ Comparison complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
