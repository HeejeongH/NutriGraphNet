#!/usr/bin/env python3
"""
make_final_summary.py
=====================
GPU 실험 SUMMARY_flat.json (또는 SUMMARY.json) 을 읽어
논문에 바로 쓸 수 있는 구조화된 SUMMARY.json을 생성합니다.

사용법:
  python make_final_summary.py                        # results/gpu/ 자동 탐색
  python make_final_summary.py path/to/SUMMARY_flat.json
  python make_final_summary.py path/to/results/gpu/  # 폴더 지정 → all_results.json 재집계

출력:
  results/FINAL_SUMMARY.json        ← 논문용 메인 파일
  results/FINAL_SUMMARY_table.txt   ← 실험별 테이블 (콘솔/메모 용)
"""

import json, sys, math
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FLAT = SCRIPT_DIR / "results" / "SUMMARY_flat_gpu.json"

# 논문에 올릴 핵심 지표 순서
PAPER_METRICS = ["auc", "f1", "HR@5", "NDCG@5", "HR@10", "NDCG@10",
                 "HR@20", "NDCG@20", "MRR",
                 "HealthGain@5", "HealthGain@10", "HealthGain@20"]

MODEL_DISPLAY = {          # 모델명 → 논문 표기
    "mf":       "MF",
    "lightgcn": "LightGCN",
    "ngcf":     "NGCF",
    "sgl":      "SGL",
    "hfrsda":   "NutriGraphNet",
    "full":     "NutriGraphNet",
}

# 실험군 분류
def group_of(exp: str) -> str:
    if exp.startswith("B_sparsity"):     return "B_sparsity"
    if exp.startswith("C_lambda"):       return "C_lambda"
    if exp.startswith("G_layers"):       return "G_layers"
    if exp.startswith("D_dim"):          return "D_dim"
    if exp.startswith("A_sgl"):          return "A_sgl_aug"
    if exp.startswith("F_ablation"):     return "F_ablation"
    return "misc"

# sparsity / lambda / layers 값 파싱
def param_of(exp: str):
    parts = exp.split("_")
    try:
        if exp.startswith("B_sparsity"):
            # B_sparsity_100pct → 100, B_sparsity_10pct → 10
            for p in parts:
                if p.endswith("pct"):
                    return int(p.replace("pct", ""))
        if exp.startswith("C_lambda"):
            return float(parts[-1])
        if exp.startswith("G_layers"):
            return int(parts[-1])
        if exp.startswith("D_dim"):
            return int(parts[-1])
        if exp.startswith("A_sgl_aug"):
            return float(parts[-1])
    except Exception:
        pass
    return exp


# ──────────────────────────────────────────────────────────────
# 지표 추출 (mean ± std)
# ──────────────────────────────────────────────────────────────
def pick_metrics(entry: dict) -> dict:
    out = {}
    for m in PAPER_METRICS:
        if m in entry:
            v = round(entry[m], 6)
            std_key = f"{m}_std"
            s = round(entry.get(std_key, 0.0), 6)
            out[m] = v
            if s > 0:
                out[std_key] = s
    return out


# ──────────────────────────────────────────────────────────────
# 통계 헬퍼
# ──────────────────────────────────────────────────────────────
def best_entry(entries: list, metric="NDCG@10") -> dict:
    return max(entries, key=lambda e: e.get("metrics", {}).get(metric, -999))

def pct_gain(base: float, improved: float) -> float:
    if base == 0:
        return float("nan")
    return round((improved - base) / abs(base) * 100, 2)


# ──────────────────────────────────────────────────────────────
# 메인 빌더
# ──────────────────────────────────────────────────────────────
def build_final_summary(flat: dict) -> dict:
    """flat = {"exp/model": {지표}} → 구조화된 논문용 dict"""

    # 1) 실험군별 그룹화
    groups = {}
    for key, entry in flat.items():
        parts = key.split("/", 1)
        if len(parts) != 2:
            continue
        exp, model = parts
        g = group_of(exp)
        groups.setdefault(g, []).append({
            "key":    key,
            "exp":    exp,
            "model":  model,
            "model_display": MODEL_DISPLAY.get(model, model),
            "param":  param_of(exp),
            "metrics": pick_metrics(entry),
        })

    # ────────────────────────────────────────────
    # 2) B: 데이터 희소성 실험
    # ────────────────────────────────────────────
    b_section = {}
    b_items = sorted(groups.get("B_sparsity", []),
                     key=lambda x: (x["param"], x["model"]))

    # sparsity 레벨별 모델 정리
    by_sparsity = {}
    for it in b_items:
        pct = it["param"]
        by_sparsity.setdefault(pct, {})[it["model_display"]] = it["metrics"]

    b_section["by_sparsity_pct"] = {
        str(pct): models for pct, models in sorted(by_sparsity.items())
    }

    # NutriGraphNet vs 베이스라인 비교 (100pct = 전체 데이터)
    full_data = by_sparsity.get(100, {})
    ng_full   = full_data.get("NutriGraphNet", {})
    baselines = {m: v for m, v in full_data.items() if m != "NutriGraphNet"}
    if ng_full and baselines:
        comparisons = {}
        for bm, bv in baselines.items():
            ng_ndcg = ng_full.get("NDCG@10", 0)
            bm_ndcg = bv.get("NDCG@10", 0)
            ng_hr   = ng_full.get("HR@10", 0)
            bm_hr   = bv.get("HR@10", 0)
            comparisons[bm] = {
                "NDCG@10_gain_pct": pct_gain(bm_ndcg, ng_ndcg),
                "HR@10_gain_pct":   pct_gain(bm_hr,   ng_hr),
            }
        b_section["nutrigraphnet_vs_baselines_100pct"] = comparisons

    # ────────────────────────────────────────────
    # 3) C: λ 민감도 실험
    # ────────────────────────────────────────────
    c_section = {}
    c_items = sorted(groups.get("C_lambda", []), key=lambda x: x["param"])
    c_section["by_lambda"] = {
        str(it["param"]): it["metrics"] for it in c_items
    }
    # 최적 λ
    best_c = best_entry(c_items, "NDCG@10")
    c_section["best_lambda"] = best_c["param"]
    c_section["best_metrics"] = best_c["metrics"]

    # HealthGain 트렌드
    hg_trend = {}
    for it in c_items:
        lam = it["param"]
        hg  = it["metrics"].get("HealthGain@10")
        nd  = it["metrics"].get("NDCG@10")
        if hg is not None:
            hg_trend[str(lam)] = {"HealthGain@10": hg, "NDCG@10": nd}
    c_section["healthgain_trend"] = hg_trend

    # ────────────────────────────────────────────
    # 4) G: GNN 레이어 수 실험
    # ────────────────────────────────────────────
    g_section = {}
    g_items = sorted(groups.get("G_layers", []),
                     key=lambda x: (x["param"], x["model"]))
    by_layer = {}
    for it in g_items:
        by_layer.setdefault(it["param"], {})[it["model_display"]] = it["metrics"]
    g_section["by_layers"] = {
        str(l): models for l, models in sorted(by_layer.items())
    }
    best_g = best_entry(g_items, "NDCG@10")
    g_section["best_layers"] = best_g["param"]
    g_section["best_model"]   = best_g["model_display"]
    g_section["best_metrics"] = best_g["metrics"]

    # ────────────────────────────────────────────
    # 5) 전체 Best 순위 (NDCG@10)
    # ────────────────────────────────────────────
    all_items = [
        {"key": k, **pick_metrics(v)}
        for k, v in flat.items()
    ]
    ranked = sorted(all_items, key=lambda x: x.get("NDCG@10", 0), reverse=True)
    top10 = ranked[:10]

    # ────────────────────────────────────────────
    # 6) 최종 구조 조합
    # ────────────────────────────────────────────
    final = {
        "_meta": {
            "description": "NutriGraphNet GPU 최종 실험 결과 (논문용)",
            "generated_at": datetime.now().isoformat(),
            "total_entries": len(flat),
            "experiments": sorted(groups.keys()),
            "paper_metrics": PAPER_METRICS,
        },

        # 실험별 상세
        "B_sparsity_experiment": b_section,
        "C_lambda_experiment":   c_section,
        "G_layers_experiment":   g_section,

        # 기타 그룹
        "other_experiments": {
            g: [
                {"key": it["key"], "param": it["param"],
                 "model": it["model_display"], "metrics": it["metrics"]}
                for it in sorted(items, key=lambda x: x["param"])
            ]
            for g, items in groups.items()
            if g not in ("B_sparsity", "C_lambda", "G_layers")
        },

        # 전체 랭킹
        "top10_by_NDCG@10": top10,

        # flat (SUMMARY_v4 호환)
        "_flat": {k: pick_metrics(v) for k, v in flat.items()},
    }

    return final


# ──────────────────────────────────────────────────────────────
# 텍스트 테이블 출력
# ──────────────────────────────────────────────────────────────
def fmt(v, ndigits=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "  N/A "
    return f"{v:.{ndigits}f}"

def make_table(summary: dict) -> str:
    lines = []
    W = 72

    def hline(c="─"): lines.append(c * W)
    def title(t):
        lines.append("=" * W)
        lines.append(f"  {t}")
        lines.append("=" * W)
    def section(t):
        hline()
        lines.append(f"  [{t}]")
        hline()

    meta = summary["_meta"]
    title("NutriGraphNet GPU 실험 최종 결과")
    lines += [
        f"  생성 시각 : {meta['generated_at']}",
        f"  총 항목   : {meta['total_entries']}",
        "",
    ]

    # ── B: 희소성 ──────────────────────────────────
    section("B. 데이터 희소성 실험")
    by_sp = summary["B_sparsity_experiment"]["by_sparsity_pct"]
    header = f"{'Sparsity':>10}  {'Model':<16}  {'AUC':>7}  {'F1':>7}  {'HR@10':>7}  {'NDCG@10':>8}  {'MRR':>7}"
    lines.append(header)
    hline("·")
    for pct, models in sorted(by_sp.items(), key=lambda x: int(x[0])):
        for model, m in sorted(models.items()):
            mark = " ★" if model == "NutriGraphNet" else ""
            lines.append(
                f"{pct+'%':>10}  {model+mark:<16}  "
                f"{fmt(m.get('auc')):>7}  {fmt(m.get('f1')):>7}  "
                f"{fmt(m.get('HR@10')):>7}  {fmt(m.get('NDCG@10')):>8}  "
                f"{fmt(m.get('MRR')):>7}"
            )
        hline("·")

    cmp = summary["B_sparsity_experiment"].get("nutrigraphnet_vs_baselines_100pct", {})
    if cmp:
        lines += ["", "  NutriGraphNet vs Baselines @ 100% 데이터 (NDCG@10 기준 향상률):"]
        for bm, vals in sorted(cmp.items()):
            nd_g = vals.get("NDCG@10_gain_pct", float("nan"))
            hr_g = vals.get("HR@10_gain_pct", float("nan"))
            lines.append(f"    vs {bm:<12}  NDCG@10 {nd_g:+.2f}%  HR@10 {hr_g:+.2f}%")
        lines.append("")

    # ── C: λ 민감도 ───────────────────────────────
    section("C. λ 민감도 실험 (NutriGraphNet full)")
    by_lam = summary["C_lambda_experiment"]["by_lambda"]
    best_lam = summary["C_lambda_experiment"]["best_lambda"]
    header = f"{'lambda':>10}  {'AUC':>7}  {'F1':>7}  {'HR@10':>7}  {'NDCG@10':>8}  {'MRR':>7}  {'HGain@10':>9}"
    lines.append(header)
    hline("·")
    for lam, m in sorted(by_lam.items(), key=lambda x: float(x[0])):
        mark = " ★" if float(lam) == best_lam else ""
        lines.append(
            f"{lam+mark:>10}  "
            f"{fmt(m.get('auc')):>7}  {fmt(m.get('f1')):>7}  "
            f"{fmt(m.get('HR@10')):>7}  {fmt(m.get('NDCG@10')):>8}  "
            f"{fmt(m.get('MRR')):>7}  {fmt(m.get('HealthGain@10'), 5):>9}"
        )
    lines += [
        "",
        f"  ★ 최적 λ = {best_lam}  "
        f"NDCG@10={fmt(summary['C_lambda_experiment']['best_metrics'].get('NDCG@10'))}  "
        f"HR@10={fmt(summary['C_lambda_experiment']['best_metrics'].get('HR@10'))}",
        "",
    ]

    # ── G: 레이어 수 ──────────────────────────────
    section("G. GNN 레이어 수 실험")
    by_lay = summary["G_layers_experiment"]["by_layers"]
    header = f"{'Layers':>8}  {'Model':<12}  {'AUC':>7}  {'F1':>7}  {'HR@10':>7}  {'NDCG@10':>8}  {'MRR':>7}"
    lines.append(header)
    hline("·")
    best_l = summary["G_layers_experiment"]["best_layers"]
    best_m = summary["G_layers_experiment"]["best_model"]
    for l, models in sorted(by_lay.items(), key=lambda x: int(x[0])):
        for model, m in sorted(models.items()):
            mark = " ★" if int(l) == best_l and model == best_m else ""
            lines.append(
                f"{l+mark:>8}  {model:<12}  "
                f"{fmt(m.get('auc')):>7}  {fmt(m.get('f1')):>7}  "
                f"{fmt(m.get('HR@10')):>7}  {fmt(m.get('NDCG@10')):>8}  "
                f"{fmt(m.get('MRR')):>7}"
            )
        hline("·")

    lines += [
        "",
        f"  ★ 최적 레이어 = {best_l}  ({best_m})  "
        f"NDCG@10={fmt(summary['G_layers_experiment']['best_metrics'].get('NDCG@10'))}",
        "",
    ]

    # ── TOP 10 ─────────────────────────────────────
    section("전체 Top-10 항목 (NDCG@10 기준)")
    header = f"{'#':>3}  {'실험/모델':<45}  {'NDCG@10':>8}  {'HR@10':>7}  {'AUC':>7}"
    lines.append(header)
    hline("·")
    for i, entry in enumerate(summary["top10_by_NDCG@10"], 1):
        lines.append(
            f"{i:>3}  {entry['key']:<45}  "
            f"{fmt(entry.get('NDCG@10')):>8}  "
            f"{fmt(entry.get('HR@10')):>7}  "
            f"{fmt(entry.get('auc')):>7}"
        )
    lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────────────────────
def main():
    # 입력 경로 결정
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FLAT

    if not src.exists():
        print(f"[ERROR] 파일/폴더 없음: {src}")
        sys.exit(1)

    # flat dict 로드
    if src.is_file():
        with open(src, encoding="utf-8") as f:
            raw = json.load(f)
        # SUMMARY.json (_flat 키) 또는 SUMMARY_flat.json (직접 flat) 모두 허용
        flat = raw.get("_flat", raw)
    else:
        # 폴더가 넘어오면 generate_summary_gpu 로직 재사용
        from generate_summary_gpu import build_summary
        s = build_summary(src)
        flat = s.get("_flat", {})

    print(f"[INFO] 로드된 항목: {len(flat)}개")
    for k in sorted(flat.keys()):
        print(f"  {k}")

    # ── 최종 SUMMARY 생성 ──────────────────────────
    final = build_final_summary(flat)

    # ── 출력 경로 ──────────────────────────────────
    out_dir = SCRIPT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # FINAL_SUMMARY.json
    out_json = out_dir / "FINAL_SUMMARY.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {out_json}")

    # FINAL_SUMMARY_table.txt
    table = make_table(final)
    out_txt = out_dir / "FINAL_SUMMARY_table.txt"
    out_txt.write_text(table, encoding="utf-8")
    print(f"[SAVED] {out_txt}")

    # 콘솔 출력
    print()
    print(table)


if __name__ == "__main__":
    main()
