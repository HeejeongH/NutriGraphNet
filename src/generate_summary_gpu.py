#!/usr/bin/env python3
"""
generate_summary_gpu.py
=======================
GPU 실험 결과 폴더에서 all_results.json을 모두 읽어 SUMMARY.json을 생성합니다.

사용법:
  # GPU 폴더가 results/gpu/ 에 있을 때
  python generate_summary_gpu.py

  # 경로 직접 지정 (Windows 경로도 OK)
  python generate_summary_gpu.py "C:/Users/.../results/gpu"

출력 파일 (GPU 폴더 내부):
  SUMMARY.json          -- 완전한 구조화 JSON (메타 + 그룹 + flat)
  SUMMARY_flat.json     -- 기존 SUMMARY_v4 호환 포맷 {"exp/model": {metrics}}
  SUMMARY_readable.txt  -- 사람이 읽기 쉬운 텍스트 보고서
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────────────────────
# 추출할 핵심 지표 (aggregated 딕셔너리 내 키)
# ──────────────────────────────────────────────────────────────
CORE_METRICS = [
    "auc", "f1",
    "accuracy", "precision", "recall",
    "ap",
    "HR@5",  "NDCG@5",
    "HR@10", "NDCG@10",
    "HR@20", "NDCG@20",
    "MRR",
    "HealthGain@5", "HealthGain@10", "HealthGain@20",
]

LOSS_KEYS = ["total", "bpr", "health", "cl"]

# 실험 이름 → 그룹 매핑
GROUP_MAP = {
    "A_": "A_sgl_augmentation",
    "B_": "B_sparsity",
    "C_": "C_lambda_sensitivity",
    "D_": "D_embedding_dim",
    "F_": "F_ablation",
    "G_": "G_gnn_layers",
}


# ──────────────────────────────────────────────────────────────
# 지표 추출
# ──────────────────────────────────────────────────────────────
def extract_metrics(aggregated: dict) -> dict:
    out = {}
    for m in CORE_METRICS:
        if m not in aggregated:
            continue
        v = aggregated[m]
        if isinstance(v, dict):
            out[m]          = round(v.get("mean", 0.0), 6)
            out[f"{m}_std"] = round(v.get("std",  0.0), 6)
        else:
            out[m] = round(float(v), 6)
    for lk in LOSS_KEYS:
        if lk not in aggregated:
            continue
        v = aggregated[lk]
        if isinstance(v, dict):
            out[f"loss_{lk}"] = round(v.get("mean", 0.0), 6)
        else:
            out[f"loss_{lk}"] = round(float(v), 6)
    return out


def load_all_results(json_path: Path) -> dict:
    """all_results.json → {model_name: metrics_dict}"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}
    for model_name, model_data in data.items():
        if not isinstance(model_data, dict):
            continue
        agg = model_data.get("aggregated")
        if agg is None:
            # 구형 포맷: 바로 지표가 최상위에 있는 경우
            agg = model_data
        metrics = extract_metrics(agg)
        if metrics:
            results[model_name] = metrics
    return results


# ──────────────────────────────────────────────────────────────
# 그룹 분류
# ──────────────────────────────────────────────────────────────
def classify_group(exp_name: str) -> str:
    for prefix, group in GROUP_MAP.items():
        if exp_name.startswith(prefix):
            return group
    return "misc"


# ──────────────────────────────────────────────────────────────
# 핵심: SUMMARY 빌드
# ──────────────────────────────────────────────────────────────
def build_summary(gpu_dir: Path) -> dict:
    flat    = {}   # {"B_sparsity_100pct/hfrsda": {metrics}}
    grouped = {}   # {"B_sparsity": {"B_sparsity_100pct": {"hfrsda": {metrics}}}}
    missing = []

    exp_dirs = sorted(
        d for d in gpu_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if not exp_dirs:
        print(f"[ERROR] 실험 폴더가 없습니다: {gpu_dir}")
        return {}

    print(f"\n{'─'*60}")
    print(f"  GPU 결과 폴더: {gpu_dir}")
    print(f"  발견된 실험 폴더: {len(exp_dirs)}개")
    print(f"{'─'*60}")

    for exp_dir in exp_dirs:
        json_path = exp_dir / "all_results.json"
        if not json_path.exists():
            missing.append(exp_dir.name)
            print(f"  [SKIP] {exp_dir.name}  (all_results.json 없음)")
            continue

        try:
            model_results = load_all_results(json_path)
        except Exception as e:
            missing.append(exp_dir.name)
            print(f"  [ERR ] {exp_dir.name}  ({e})")
            continue

        if not model_results:
            missing.append(exp_dir.name)
            print(f"  [WARN] {exp_dir.name}  (지표 없음)")
            continue

        group = classify_group(exp_dir.name)
        grouped.setdefault(group, {}).setdefault(exp_dir.name, {})

        for model_name, metrics in model_results.items():
            key = f"{exp_dir.name}/{model_name}"
            flat[key] = metrics
            grouped[group][exp_dir.name][model_name] = metrics
            print(f"  [OK ] {key:<55s}  ({len(metrics)} metrics)")

    print(f"\n  ✓ 처리 완료: {len(flat)}개 항목")
    if missing:
        print(f"  ✗ 누락 폴더: {missing}")

    meta = {
        "description": "NutriGraphNet GPU 실험 결과 요약 (자동 생성)",
        "generated_at": datetime.now().isoformat(),
        "source_dir": str(gpu_dir),
        "total_entries": len(flat),
        "groups": sorted(grouped.keys()),
        "missing_folders": missing,
    }

    return {"_meta": meta, "_flat": flat, "_grouped": grouped}


# ──────────────────────────────────────────────────────────────
# 텍스트 보고서
# ──────────────────────────────────────────────────────────────
def write_readable(summary: dict, out_path: Path):
    meta    = summary.get("_meta", {})
    grouped = summary.get("_grouped", {})

    lines = []
    lines += [
        "=" * 70,
        "  NutriGraphNet GPU 실험 결과 — SUMMARY_readable.txt",
        "=" * 70,
        f"  생성 시각  : {meta.get('generated_at', 'N/A')}",
        f"  소스 폴더  : {meta.get('source_dir', 'N/A')}",
        f"  전체 항목  : {meta.get('total_entries', 0)}개",
        f"  실험군     : {', '.join(meta.get('groups', []))}",
        "",
    ]

    for group_name, exps in sorted(grouped.items()):
        lines += ["─" * 70, f"[{group_name}]", "─" * 70, ""]
        for exp_name, models in sorted(exps.items()):
            lines.append(f"  {exp_name}")
            for model_name, metrics in sorted(models.items()):
                # 핵심 지표만 한 줄 표시
                keys = ["auc", "f1", "HR@10", "NDCG@10", "MRR"]
                parts = []
                for k in keys:
                    if k in metrics:
                        parts.append(f"{k}={metrics[k]:.4f}")
                # HealthGain 있으면 추가
                for hg in ["HealthGain@10"]:
                    if hg in metrics:
                        parts.append(f"{hg}={metrics[hg]:.4f}")
                lines.append(f"    ├─ {model_name:<15s}  {' | '.join(parts)}")
            lines.append("")

    # 그룹별 최고 성능
    lines += ["─" * 70, "[Best per Group  (NDCG@10 기준)]", "─" * 70, ""]
    flat = summary.get("_flat", {})
    best_per = {}
    for key, m in flat.items():
        g = classify_group(key.split("/")[0])
        if g not in best_per or m.get("NDCG@10", 0) > best_per[g]["NDCG@10"]:
            best_per[g] = {"key": key, **m}
    for g, info in sorted(best_per.items()):
        k   = info["key"]
        ndcg = info.get("NDCG@10", 0)
        hr   = info.get("HR@10", 0)
        auc  = info.get("auc", 0)
        lines.append(f"  {g:<30s}  {k:<50s}  NDCG@10={ndcg:.4f}  HR@10={hr:.4f}  AUC={auc:.4f}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  → 텍스트 보고서: {out_path.name}")


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
def main():
    # 경로 결정
    if len(sys.argv) > 1:
        gpu_dir = Path(sys.argv[1])
    else:
        gpu_dir = Path(__file__).resolve().parent / "results" / "gpu"

    if not gpu_dir.exists():
        print(f"[ERROR] 폴더 없음: {gpu_dir}")
        print()
        print("  다음 중 하나를 수행하세요:")
        print("  1) results/gpu/ 폴더를 만들고 실험 결과를 복사")
        print("  2) 직접 경로 지정: python generate_summary_gpu.py <경로>")
        print()
        print("  예) python generate_summary_gpu.py \"C:/Users/.../results/gpu\"")
        sys.exit(1)

    # 빌드
    summary = build_summary(gpu_dir)
    if not summary:
        print("[ERROR] 결과 없음, 종료합니다.")
        sys.exit(1)

    # ── SUMMARY.json 저장 ────────────────────────
    out_json = gpu_dir / "SUMMARY.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  → SUMMARY.json 저장: {out_json}")

    # ── SUMMARY_flat.json (v4 호환) 저장 ────────
    flat_json = gpu_dir / "SUMMARY_flat.json"
    with open(flat_json, "w", encoding="utf-8") as f:
        json.dump(summary["_flat"], f, ensure_ascii=False, indent=2)
    print(f"  → SUMMARY_flat.json 저장: {flat_json}")

    # ── 텍스트 보고서 저장 ───────────────────────
    write_readable(summary, gpu_dir / "SUMMARY_readable.txt")

    # ── 콘솔에 최고 성능 요약 출력 ──────────────
    print("\n" + "=" * 70)
    print("  Best per Group (NDCG@10 기준)")
    print("=" * 70)
    flat = summary["_flat"]
    best_per = {}
    for key, m in flat.items():
        g = classify_group(key.split("/")[0])
        if g not in best_per or m.get("NDCG@10", 0) > best_per[g]["NDCG@10"]:
            best_per[g] = {"key": key, **m}
    for g, info in sorted(best_per.items()):
        k    = info["key"]
        ndcg = info.get("NDCG@10", 0)
        hr   = info.get("HR@10", 0)
        auc  = info.get("auc", 0)
        print(f"  {g:<28s} → {k:<48s}  NDCG@10={ndcg:.4f}  HR@10={hr:.4f}  AUC={auc:.4f}")

    print("\n✅  SUMMARY 생성 완료!")
    print(f"    Upload this file to update the paper: {out_json}")


if __name__ == "__main__":
    main()
