"""
Compare results from multiple experiments
"""

import json
import os
from pathlib import Path
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_results(results_dir='results'):
    """Load all result JSON files"""
    results = []
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return results
    
    json_files = list(results_path.glob('*_results.json'))
    
    if not json_files:
        print(f"❌ No result files found in {results_dir}")
        return results
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            result = json.load(f)
            results.append(result)
    
    return results

def print_comparison_table(results):
    """Print comparison table"""
    if not results:
        print("No results to display")
        return
    
    table_data = []
    
    for result in results:
        row = [
            result['model'],
            result['loss'],
            f"{result['final_metrics']['f1']:.4f}",
            f"{result['final_metrics']['auc']:.4f}",
            f"{result['final_metrics']['accuracy']:.4f}",
            f"{result['final_metrics']['precision']:.4f}",
            f"{result['final_metrics']['recall']:.4f}",
            f"{result['total_time']/60:.1f} min"
        ]
        table_data.append(row)
    
    # Sort by F1 score (descending)
    table_data.sort(key=lambda x: float(x[2]), reverse=True)
    
    headers = ['Model', 'Loss', 'F1', 'AUC', 'Accuracy', 'Precision', 'Recall', 'Time']
    
    print("\n" + "="*100)
    print("📊 EXPERIMENT RESULTS COMPARISON")
    print("="*100 + "\n")
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Find best model
    best_model = table_data[0]
    print(f"\n🏆 Best Model: {best_model[0]} ({best_model[1]} loss)")
    print(f"   F1 Score: {best_model[2]}")
    print(f"   AUC: {best_model[3]}")
    print(f"   Training Time: {best_model[7]}")

def plot_results(results):
    """Plot comparison charts"""
    if not results:
        return
    
    # Prepare data
    models = [f"{r['model']}\n({r['loss']})" for r in results]
    f1_scores = [r['final_metrics']['f1'] for r in results]
    auc_scores = [r['final_metrics']['auc'] for r in results]
    
    # Sort by F1 score
    sorted_indices = np.argsort(f1_scores)[::-1]
    models = [models[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    auc_scores = [auc_scores[i] for i in sorted_indices]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # F1 Score comparison
    axes[0].barh(models, f1_scores, color='steelblue')
    axes[0].set_xlabel('F1 Score', fontsize=12)
    axes[0].set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0, 1)
    axes[0].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(f1_scores):
        axes[0].text(v + 0.01, i, f'{v:.4f}', va='center')
    
    # AUC comparison
    axes[1].barh(models, auc_scores, color='coral')
    axes[1].set_xlabel('AUC', fontsize=12)
    axes[1].set_title('AUC Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, 1)
    axes[1].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(auc_scores):
        axes[1].text(v + 0.01, i, f'{v:.4f}', va='center')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('results/comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 Comparison chart saved: results/comparison.png")
    
    plt.show()

def export_to_csv(results):
    """Export results to CSV"""
    if not results:
        return
    
    import csv
    
    csv_path = 'results/comparison.csv'
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Model', 'Loss', 'F1', 'AUC', 'Accuracy', 
            'Precision', 'Recall', 'Time(min)'
        ])
        
        # Data
        for result in results:
            writer.writerow([
                result['model'],
                result['loss'],
                f"{result['final_metrics']['f1']:.4f}",
                f"{result['final_metrics']['auc']:.4f}",
                f"{result['final_metrics']['accuracy']:.4f}",
                f"{result['final_metrics']['precision']:.4f}",
                f"{result['final_metrics']['recall']:.4f}",
                f"{result['total_time']/60:.1f}"
            ])
    
    print(f"📄 Results exported to CSV: {csv_path}")

def main():
    """Main function"""
    print("🔍 Loading experiment results...")
    
    results = load_results('results')
    
    if not results:
        print("\n⚠️ No results found. Please run experiments first:")
        print("   bash run_all_experiments.sh")
        return
    
    print(f"✅ Loaded {len(results)} experiment results\n")
    
    # Print table
    print_comparison_table(results)
    
    # Plot charts
    try:
        plot_results(results)
    except Exception as e:
        print(f"\n⚠️ Could not generate plots: {e}")
        print("   Install matplotlib and seaborn: pip install matplotlib seaborn")
    
    # Export to CSV
    export_to_csv(results)
    
    print("\n" + "="*100)
    print("✅ Comparison complete!")
    print("="*100 + "\n")

if __name__ == '__main__':
    main()
