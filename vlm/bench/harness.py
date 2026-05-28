# =============================================================================
# vlm/bench/harness.py
# -----------------------------------------------------------------------------
# Eval harness 단일 진입점.
# vlm/bench/registry.yaml 의 모델/평가셋/회귀 임계치 선언을 읽어
# 일괄 추론·스코어·회귀 검사를 수행한다.
#
# 사용 예:
#   # 등록된 모든 모델 일괄 추론 (legacy_results 가 있으면 skip)
#   python -m vlm.bench.harness run
#
#   # 특정 모델만 (재학습 후 새 어댑터 검증)
#   python -m vlm.bench.harness run --models lora_v5
#
#   # 강제 재추론 (legacy 무시)
#   python -m vlm.bench.harness run --models lora_v4 --force
#
#   # 모든 등록 모델 스코어 + N-way 리포트 (기본: legacy 우선, 없으면 최신 run)
#   python -m vlm.bench.harness score
#
#   # 회귀 검사 — baseline 대비 임계치 위반 시 exit 1
#   python -m vlm.bench.harness check --candidate lora_v4
#
# 출력:
#   vlm/bench/runs/<UTC-timestamp>__<git-sha>__<label>/
#     ├── manifest.json   (git_sha, adapter_sha256, eval_hash, env, timing)
#     └── results.jsonl   (runner.run() 결과)
#   vlm/bench/score_report.md     (N-way 비교 리포트 — score 시 갱신)
#   vlm/bench/regression.json     (check 시 위반 상세)
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from vlm.bench import runner as bench_runner
from vlm.bench import scorer as bench_scorer

REGISTRY_PATH = ROOT / "vlm" / "bench" / "registry.yaml"
RUNS_DIR      = ROOT / "vlm" / "bench" / "runs"
REPORT_PATH   = ROOT / "vlm" / "bench" / "score_report.md"
REGRESSION_PATH = ROOT / "vlm" / "bench" / "regression.json"
TREND_PATH    = ROOT / "vlm" / "bench" / "score_trend.md"


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode())


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"registry not found: {path}")
    with open(path, encoding="utf-8") as f:
        reg = yaml.safe_load(f)
    labels = [m["label"] for m in reg.get("models", [])]
    if len(labels) != len(set(labels)):
        raise ValueError(f"registry: duplicate model labels — {labels}")
    return reg


def git_sha(short: bool = True) -> str:
    try:
        flag = "--short" if short else "HEAD"
        if short:
            out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT)
        else:
            out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        return out.decode().strip()
    except Exception:
        return "unknown"


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def adapter_fingerprint(adapter_dir: str | None) -> dict | None:
    if not adapter_dir:
        return None
    p = (ROOT / adapter_dir).resolve() if not Path(adapter_dir).is_absolute() else Path(adapter_dir)
    weights = p / "adapter_model.safetensors"
    cfg     = p / "adapter_config.json"
    return {
        "dir":                        str(p),
        "exists":                     weights.exists(),
        "adapter_model_sha256":       file_sha256(weights),
        "adapter_config_sha256":      file_sha256(cfg),
    }


def env_fingerprint() -> dict:
    info: dict = {
        "python":   sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in ("torch", "transformers", "peft", "numpy"):
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            info[pkg] = "missing"
    return info


def eval_set_hash(path: Path) -> str | None:
    return file_sha256(path)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ── run 서브커맨드 ───────────────────────────────────────────────────────────

def cmd_run(args, reg: dict):
    selected = _select_models(reg, args.models)
    eval_cfg = reg["eval_set"]
    eval_path = ROOT / eval_cfg["path"]

    sha = git_sha(short=True)
    env = env_fingerprint()
    started = utc_stamp()

    for model in selected:
        label = model["label"]
        legacy = model.get("legacy_results")

        if legacy and not args.force:
            safe_print(f"[run] {label}: legacy_results 보유 ({legacy}) — skip (--force 로 재추론)")
            continue

        run_dir = RUNS_DIR / f"{started}__{sha}__{label}"
        run_dir.mkdir(parents=True, exist_ok=True)
        results_path = run_dir / "results.jsonl"

        safe_print(f"[run] {label} → {results_path}")
        t0 = time.time()
        try:
            bench_runner.run(
                mode=model["mode"],
                n=eval_cfg["n"],
                seed=eval_cfg["seed"],
                output=results_path,
                adapter_path=model.get("adapter_path"),
                tag=label,
            )
            err = None
        except Exception as e:
            err = repr(e)
            safe_print(f"[run] {label} FAILED: {err}")
        elapsed = round(time.time() - t0, 2)

        manifest = {
            "label":          label,
            "model_mode":     model["mode"],
            "adapter":        adapter_fingerprint(model.get("adapter_path")),
            "eval_set": {
                "source":     eval_cfg["source"],
                "n":          eval_cfg["n"],
                "seed":       eval_cfg["seed"],
                "path":       str(eval_path),
                "sha256":     eval_set_hash(eval_path),
            },
            "git_sha":        git_sha(short=False),
            "env":            env,
            "started_utc":    started,
            "elapsed_sec":    elapsed,
            "error":          err,
            "results_path":   str(results_path),
        }
        with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        safe_print(f"[run] {label} done in {elapsed}s")


# ── score 서브커맨드 ─────────────────────────────────────────────────────────

def cmd_score(args, reg: dict):
    inputs = _resolve_inputs(reg, args.models, prefer_legacy=not args.prefer_runs)
    if not inputs:
        safe_print("[score] 비교할 결과가 없습니다.")
        sys.exit(2)

    baseline_label = args.baseline or reg.get("baseline") or next(iter(inputs))
    if baseline_label not in inputs:
        safe_print(f"[score] baseline '{baseline_label}' 결과 없음, 첫 항목으로 대체")
        baseline_label = next(iter(inputs))

    metrics_dict: dict[str, dict] = {}
    for label, path in inputs.items():
        records = bench_scorer.load_jsonl(path)
        safe_print(f"[score] {label}: {len(records)}건  ({path})")
        metrics_dict[label] = bench_scorer.evaluate(records)

    bench_scorer.write_report(metrics_dict, REPORT_PATH, baseline_key=baseline_label)
    safe_print(f"[score] 리포트: {REPORT_PATH}")
    safe_print(f"[score] baseline = {baseline_label}")

    if args.check:
        regs = _check_regression(metrics_dict, baseline_label, reg.get("regression", {}))
        _print_regression(regs)
        _write_regression(regs, baseline_label, metrics_dict)
        if any(r["violated"] for r in regs):
            sys.exit(1)


# ── check 서브커맨드 ────────────────────────────────────────────────────────

def cmd_check(args, reg: dict):
    if not args.candidate:
        safe_print("[check] --candidate <label> 필요")
        sys.exit(2)
    labels = [args.baseline or reg.get("baseline"), args.candidate]
    inputs = _resolve_inputs(reg, labels, prefer_legacy=not args.prefer_runs)
    baseline_label = args.baseline or reg.get("baseline")

    if baseline_label not in inputs or args.candidate not in inputs:
        safe_print(f"[check] 결과 누락 — baseline={baseline_label in inputs}, candidate={args.candidate in inputs}")
        sys.exit(2)

    metrics_dict = {lbl: bench_scorer.evaluate(bench_scorer.load_jsonl(p)) for lbl, p in inputs.items()}
    regs = _check_regression(metrics_dict, baseline_label, reg.get("regression", {}))
    _print_regression(regs)
    _write_regression(regs, baseline_label, metrics_dict)
    if any(r["violated"] for r in regs):
        sys.exit(1)


# ── trend 서브커맨드 ────────────────────────────────────────────────────────

def cmd_trend(args, reg: dict):
    """runs/ 디렉터리의 모든 실행을 시간순으로 모아 per-label metric 추이 작성."""
    if not RUNS_DIR.exists():
        safe_print(f"[trend] runs 디렉터리 없음: {RUNS_DIR}")
        sys.exit(2)

    entries: list[dict] = []
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest_p = d / "manifest.json"
        results_p  = d / "results.jsonl"
        if not (manifest_p.exists() and results_p.exists()):
            continue
        try:
            with open(manifest_p, encoding="utf-8") as f:
                man = json.load(f)
            records = bench_scorer.load_jsonl(results_p)
            m = bench_scorer.evaluate(records)
        except Exception as e:
            safe_print(f"[trend] {d.name} 스킵: {e}")
            continue
        entries.append({
            "run_dir":   d.name,
            "label":     man.get("label"),
            "started":   man.get("started_utc"),
            "git_sha":   (man.get("git_sha") or "")[:7],
            "metrics":   m,
        })

    if not entries:
        safe_print("[trend] 유효한 run 없음. `harness run` 으로 실행 결과를 먼저 생성하세요.")
        sys.exit(2)

    # 라벨별로 group → 시간순 정렬
    by_label: dict[str, list[dict]] = {}
    for e in entries:
        by_label.setdefault(e["label"], []).append(e)
    for lbl in by_label:
        by_label[lbl].sort(key=lambda x: x["started"] or "")

    track_keys = [
        ("rouge_l",         "ROUGE-L"),
        ("rouge_l_max",     "ROUGE-L max"),
        ("bert_score_f1",   "BERT F1"),
        ("distinct_2",      "Distinct-2"),
        ("grade_match_rate","Grade 일치율"),
        ("elapsed_avg_sec", "Avg 추론(초)"),
    ]

    with open(TREND_PATH, "w", encoding="utf-8") as f:
        f.write("# Eval Harness — 시간순 metric 추이\n\n")
        f.write(f"총 {len(entries)}개 run, {len(by_label)}개 라벨\n\n")
        for lbl in sorted(by_label):
            runs = by_label[lbl]
            f.write(f"## `{lbl}` — {len(runs)}회 실행\n\n")
            f.write("| started_utc | git | " + " | ".join(name for _, name in track_keys) + " |\n")
            f.write("|" + "|".join(["---"] * (2 + len(track_keys))) + "|\n")
            prev: dict | None = None
            for r in runs:
                m = r["metrics"]
                row = [r["started"] or "-", r["git_sha"] or "-"]
                for k, _ in track_keys:
                    v = m.get(k, -1.0)
                    cell = f"{v:.4f}" if isinstance(v, (int, float)) and v != -1.0 else "N/A"
                    if prev is not None:
                        pv = prev.get(k, -1.0)
                        if isinstance(v, (int, float)) and isinstance(pv, (int, float)) and pv not in (0, -1.0) and v != -1.0:
                            delta = (v - pv) / pv * 100
                            cell += f" ({delta:+.1f}%)"
                    row.append(cell)
                f.write("| " + " | ".join(row) + " |\n")
                prev = m
            f.write("\n")

    safe_print(f"[trend] {TREND_PATH} ({len(entries)} runs)")


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _select_models(reg: dict, names: list[str] | None) -> list[dict]:
    all_models = reg["models"]
    if not names:
        return all_models
    by_label = {m["label"]: m for m in all_models}
    missing = [n for n in names if n not in by_label]
    if missing:
        raise SystemExit(f"unknown model label(s): {missing}. registered: {list(by_label)}")
    return [by_label[n] for n in names]


def _resolve_inputs(reg: dict, names: list[str] | None, prefer_legacy: bool) -> dict[str, Path]:
    """label → results.jsonl 경로 매핑. 우선순위:
       - prefer_legacy=True: legacy_results 우선, 없으면 가장 최근 run
       - prefer_legacy=False: 가장 최근 run 우선, 없으면 legacy
    """
    selected = _select_models(reg, names)
    out: dict[str, Path] = {}
    for m in selected:
        label = m["label"]
        legacy = ROOT / m["legacy_results"] if m.get("legacy_results") else None
        recent = _find_latest_run(label)
        cand_order = (legacy, recent) if prefer_legacy else (recent, legacy)
        chosen = next((p for p in cand_order if p and p.exists()), None)
        if chosen:
            out[label] = chosen
    return out


def _find_latest_run(label: str) -> Path | None:
    if not RUNS_DIR.exists():
        return None
    matches = sorted(
        (d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.endswith(f"__{label}")),
        reverse=True,
    )
    for d in matches:
        p = d / "results.jsonl"
        if p.exists():
            return p
    return None


def _check_regression(metrics: dict[str, dict], baseline_label: str, rules: dict) -> list[dict]:
    base = metrics[baseline_label]
    out = []
    for cand_label, m in metrics.items():
        if cand_label == baseline_label:
            continue
        for metric, rule in rules.items():
            b, x = base.get(metric), m.get(metric)
            if b in (None, -1.0, -1) or x in (None, -1.0, -1) or b == 0:
                continue
            delta_pct = (x - b) / b * 100
            min_d = rule.get("min_delta_pct")
            max_d = rule.get("max_delta_pct")
            violated = False
            reason = ""
            if min_d is not None and delta_pct < min_d:
                violated = True
                reason = f"{delta_pct:+.1f}% < min {min_d:+.1f}%"
            if max_d is not None and delta_pct > max_d:
                violated = True
                reason = f"{delta_pct:+.1f}% > max {max_d:+.1f}%"
            out.append({
                "candidate":  cand_label,
                "metric":     metric,
                "baseline":   round(b, 4),
                "value":      round(x, 4),
                "delta_pct":  round(delta_pct, 2),
                "rule":       rule,
                "violated":   violated,
                "reason":     reason,
            })
    return out


def _print_regression(regs: list[dict]):
    if not regs:
        safe_print("[check] 회귀 임계치 정의 없음 또는 비교 대상 없음")
        return
    fails = [r for r in regs if r["violated"]]
    safe_print(f"[check] {len(regs)}개 비교 / 위반 {len(fails)}건")
    for r in regs:
        mark = "FAIL" if r["violated"] else " ok "
        safe_print(f"  [{mark}] {r['candidate']:25s} {r['metric']:18s} "
                   f"base={r['baseline']:.4f}  val={r['value']:.4f}  Δ={r['delta_pct']:+.1f}%  {r['reason']}")


def _write_regression(regs: list[dict], baseline_label: str, metrics: dict[str, dict]):
    payload = {
        "baseline":  baseline_label,
        "checked_utc": utc_stamp(),
        "regressions": regs,
        "metrics": metrics,
    }
    with open(REGRESSION_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    safe_print(f"[check] 상세 저장: {REGRESSION_PATH}")


# ── 진입점 ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="VLM eval harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="등록 모델 추론")
    pr.add_argument("--models", nargs="*", default=None, help="라벨 목록 (생략=전체)")
    pr.add_argument("--force",  action="store_true", help="legacy_results 무시하고 재추론")

    ps = sub.add_parser("score", help="N-way 스코어 + 리포트")
    ps.add_argument("--models", nargs="*", default=None)
    ps.add_argument("--baseline", type=str, default=None)
    ps.add_argument("--prefer-runs", action="store_true", help="legacy 보다 최신 run 우선")
    ps.add_argument("--check", action="store_true", help="회귀 임계치도 함께 검사 (위반 시 exit 1)")

    pc = sub.add_parser("check", help="회귀 검사 (baseline vs candidate)")
    pc.add_argument("--candidate", type=str, required=True)
    pc.add_argument("--baseline",  type=str, default=None)
    pc.add_argument("--prefer-runs", action="store_true")

    pt = sub.add_parser("trend", help="runs/ 의 모든 실행을 시간순 metric 추이로 정리")

    args = p.parse_args()
    reg = load_registry()

    if args.cmd == "run":
        cmd_run(args, reg)
    elif args.cmd == "score":
        cmd_score(args, reg)
    elif args.cmd == "check":
        cmd_check(args, reg)
    elif args.cmd == "trend":
        cmd_trend(args, reg)


if __name__ == "__main__":
    main()
