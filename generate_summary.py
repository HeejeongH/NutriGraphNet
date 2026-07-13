#!/usr/bin/env python3
"""
generate_summary.py
===================
GPU 실험 결과 폴더(results/gpu/)에 있는 모든 all_results.json을 읽어
단일 SUMMARY.json으로 집계합니다.

폴더 구조:
  results/gpu/
    B_sparsity_10pct/all_results.json   -> 모델별(mf, lightgcn, ngcf, sgl, hfrsda)
    B_sparsity_30pct/all_results.json
    B_sparsity_50pct/all_results.json
    B_sparsity_70pct/all_results.json
    B_sparsity_100pct/all_results.json
    C_lambda_0.0/all_results.json       -> 모델별(full)
    C_lambda_0.001/all_results.json
    ...
    G_layers_1/all_results.json         -> 모델별(lightgcn, ngcf)
    G_layers_2/all_results.json
    G_layers_3/all_results.json
    G_layers_4/all_results.json

출력:
  results/gpu/SUMMARY.json
  results/gpu/SUMMARY_readable.txt  (사람이 읽기 쉬운 텍스트)
"""

import json
import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
# 스크립트가 NutriGraphNet 폴더 또는 webapp 폴더에서 실행 가능
SCRIPT_DIR = Path(__file__).resolve().parent
GPU_RESULTS_DIR = SCRIPT_DIR / "results" / "gpu"

# 추출할 핵심 지표 (aggregated 아래 필드명)
CORE_METRICS = [
    "auc",
    "f1",
    "HR@5", "NDCG@5",
    "HR@10", "NDCG@10",
    "HR@20", "NDCG@20",
    "MRR",
    "HealthGain@5", "HealthGain@10", "HealthGain@20",
]


# ─────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────
def extract_metrics(aggregated: dict) -> dict:
    """aggregated 딕셔너리에서 핵심 지표의 mean/std를 추출합니다."""
    out = {}
    for metric in CORE_METRICS:
        if metric in aggregated:
            v = aggregated[metric]
            if isinstance(v, dict):
                out[metric] = round(v.get("mean", 0.0), 6)
                out[f"{metric}_std"] = round(v.get("std", 0.0), 6)
            else:
                out[metric] = round(float(v), 6)
    # loss 계열도 포함
    for loss_key in ["total", "bpr", "health", "cl"]:
        if loss_key in aggregated:
            v = aggregated[loss_key]
            if isinstance(v, dict):
                out[f"loss_{loss_key}"] = round(v.get("mean", 0.0), 6)
    return out


def load_all_results(json_path: Path) -> dict:
    """all_results.json 파일을 로드하여 모델명 -> 지표 딕셔너리 반환."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}
    for model_name, model_data in data.items():
        if not isinstance(model_data, dict):
            continue
        agg = model_data.get("aggregated", {})
        if not agg:
            # 구형 포맷: 직접 지표가 최상위에 있을 수 있음
            agg = model_data
        metrics = extract_metrics(agg)
        if metrics:
            results[model_name] = metrics
    return results


# ─────────────────────────────────────────────
# 메인 로직
# ─────────────────────────────────────────────
def build_summary(gpu_dir: Path) -> dict:
    """GPU 결과 폴더를 순회하여 SUMMARY 딕셔너리 생성."""
    summary = {}
    missing = []
    processed = []

    # 실험 폴더 패턴 정렬
    experiment_dirs = sorted([
        d for d in gpu_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not experiment_dirs:
        print(f"[ERROR] 실험 폴더를 찾을 수 없습니다: {gpu_dir}")
        return summary

    print(f"[INFO] 발견된 실험 폴더: {len(experiment_dirs)}개")

    for exp_dir in experiment_dirs:
        json_path = exp_dir / "all_results.json"
        if not json_path.exists():
            missing.append(str(exp_dir.name))
            continue

        try:
            model_results = load_all_results(json_path)
        except Exception as e:
            print(f"[WARN] {exp_dir.name}/all_results.json 파싱 실패: {e}")
            missing.append(str(exp_dir.name))
            continue

        if not model_results:
            print(f"[WARN] {exp_dir.name}: 지표 없음")
            missing.append(str(exp_dir.name))
            continue

        for model_name, metrics in model_results.items():
            key = f"{exp_dir.name}/{model_name}"
            summary[key] = metrics
            processed.append(key)
            print(f"  [OK] {key}  ({len(metrics)} 지표)")

    print(f"\n[INFO] 처리 완료: {len(processed)}개 항목")
    if missing:
        print(f"[WARN] all_results.json 없는 폴더 ({len(missing)}개): {missing}")

    return summary


def build_summary_with_meta(gpu_dir: Path) -> dict:
    """실험 메타정보를 포함한 구조화된 SUMMARY 생성."""
    raw = build_summary(gpu_dir)

    # ── 실험군별로 그룹화 ──────────────────────────
    groups = {}
    for key, metrics in raw.items():
        # key 예: "B_sparsity_100pct/hfrsda"
        parts = key.split("/")
        if len(parts) >= 2:
            exp_name = parts[0]
            model_name = "/".join(parts[1:])
        else:
            exp_name = key
            model_name = "unknown"

        # 실험군 분류
        if exp_name.startswith("A_"):
            group = "A_sgl_augmentation"
        elif exp_name.startswith("B_"):
            group = "B_sparsity"
        elif exp_name.startswith("C_"):
            group = "C_lambda_sensitivity"
        elif exp_name.startswith("D_"):
            group = "D_embedding_dim"
        elif exp_name.startswith("F_"):
            group = "F_ablation"
        elif exp_name.startswith("G_"):
            group = "G_gnn_layers"
        else:
            group = "misc"

        if group not in groups:
            groups[group] = {}
        if exp_name not in groups[group]:
            groups[group][exp_name] = {}
        groups[group][exp_name][model_name] = metrics

    # ── 최종 구조 ──────────────────────────────────
    final = {
        "_meta": {
            "description": "NutriGraphNet GPU 실험 결과 요약",
            "generated": __import__("datetime").datetime.now().isoformat(),
            "source_dir": str(gpu_dir),
            "total_entries": len(raw),
            "groups": list(groups.keys()),
        },
        "_flat": raw,          # 기존 SUMMARY_v4 호환 포맷 (key: "exp/model")
        "_grouped": groups,    # 실험군별 계층 구조
    }
    return final


def write_readable_txt(summary: dict, out_path: Path):
    """사람이 읽기 쉬운 텍스트 보고서 생성."""
    lines = []
    lines.append("=" * 70)
    lines.append("  NutriGraphNet GPU 실험 결과 요약 (SUMMARY_readable.txt)")
    lines.append("=" * 70)

    meta = summary.get("_meta", {})
    lines.append(f"생성 시각: {meta.get('generated', 'N/A')}")
    lines.append(f"전체 항목: {meta.get('total_entries', 0)}개")
    lines.append(f"실험군: {', '.join(meta.get('groups', []))}")
    lines.append("")

    grouped = summary.get("_grouped", {})

    for group_name, exps in sorted(grouped.items()):
        lines.append("─" * 70)
        lines.append(f"[{group_name}]")
        lines.append("─" * 70)

        for exp_name, models in sorted(exps.items()):
            lines.append(f"\n  실험: {exp_name}")
            for model_name, metrics in sorted(models.items()):
                lines.append(f"    └── {model_name}")
                # 핵심 지표만 표시
                display = {k: v for k, v in metrics.items()
                           if not k.endswith("_std") and not k.startswith("loss_")}
                metric_strs = []
                for mk, mv in display.items():
                    metric_strs.append(f"{mk}={mv:.4f}")
                lines.append("         " + " | ".join(metric_strs))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] 텍스트 보고서 저장: {out_path}")


def find_best_per_group(summary: dict) -> dict:
    """각 실험군에서 NDCG@10 기준 최고 성능 항목을 찾아 반환."""
    flat = summary.get("_flat", {})
    best = {}

    for key, metrics in flat.items():
        ndcg = metrics.get("NDCG@10", 0.0)
        exp_name = key.split("/")[0]
        # 실험군 prefix
        group = exp_name[:1] + "_" if exp_name else "?"

        if group not in best or ndcg > best[group]["NDCG@10"]:
            best[group] = {"key": key, "NDCG@10": ndcg, **metrics}

    return best


# ─────────────────────────────────────────────
# 엔트리포인트
# ─────────────────────────────────────────────
def main():
    # 경로 재정의 (명령줄 인수로 GPU 폴더 지정 가능)
    if len(sys.argv) > 1:
        gpu_dir = Path(sys.argv[1])
    else:
        gpu_dir = GPU_RESULTS_DIR

    if not gpu_dir.exists():
        print(f"[ERROR] GPU 결과 폴더가 없습니다: {gpu_dir}")
        print("사용법: python generate_summary.py [results/gpu 경로]")
        sys.exit(1)

    print(f"[INFO] GPU 결과 폴더: {gpu_dir}")
    print()

    # ── 요약 생성 ──────────────────────────────
    summary = build_summary_with_meta(gpu_dir)

    # ── SUMMARY.json 저장 ──────────────────────
    out_json = gpu_dir / "SUMMARY.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] SUMMARY.json 저장: {out_json}")

    # ── 기존 SUMMARY_v4 호환 포맷도 별도 저장 ──
    flat_path = gpu_dir / "SUMMARY_flat.json"
    with open(flat_path, "w", encoding="utf-8") as f:
        json.dump(summary["_flat"], f, ensure_ascii=False, indent=2)
    print(f"[INFO] SUMMARY_flat.json (v4 호환) 저장: {flat_path}")

    # ── 텍스트 보고서 ──────────────────────────
    txt_path = gpu_dir / "SUMMARY_readable.txt"
    write_readable_txt(summary, txt_path)

    # ── 그룹별 최고 성능 출력 ──────────────────
    print("\n[BEST per group - NDCG@10 기준]")
    best = find_best_per_group(summary)
    for group, info in sorted(best.items()):
        key = info.pop("key")
        ndcg = info.get("NDCG@10", 0.0)
        hr10 = info.get("HR@10", 0.0)
        auc  = info.get("auc", 0.0)
        print(f"  {group} → {key:50s}  NDCG@10={ndcg:.4f}  HR@10={hr10:.4f}  AUC={auc:.4f}")

    print("\n완료!")
    return out_json


if __name__ == "__main__":
    main()
