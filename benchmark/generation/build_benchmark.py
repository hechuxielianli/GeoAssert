"""Assemble the experimental controlled-failure evaluation set.

``scale`` mode downloads reference programs, synthesizes assertions, creates
mutations, filters them, and writes JSONL plus metadata. ``augment`` mode adds
feedback payloads and difficulty metadata to an existing gated JSONL file.

Examples:
  python build_benchmark.py --smoke --no-symmetry
  python build_benchmark.py --mode augment --inputs gated.jsonl --out bench.jsonl
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THIS_FILE = Path(__file__).resolve()
GENERATION_DIR = Path(__file__).resolve().parent
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))
from geoassert import build_repair_prompt, check, split_steps
from geoassert.constants import SPLIT_TEST_THRESHOLD
from geoassert.metrics import (
    chamfer_distance_points, code_seed, gold_reference, sample_points,
)
from inject_errors import ERROR_CLASSES as INJECT_CLASSES  # noqa: E402
from inject_errors import inject, run_gold_props_cached  # noqa: E402
from synthesize_assertions import synthesize  # noqa: E402

RESULT_SENTINEL = b"@@GEOASSERT_EVAL_RESULT@@"
DEFAULT_STEP_TIMEOUT = 30.0
DEFAULT_WALL_TIMEOUT = 180.0
DEFAULT_N_POINTS = 2048
ERROR_CLASSES = INJECT_CLASSES
BOOL_MARKERS = (".cut(", ".union(", ".intersect(", ".cut (")
HF_REPO = "gudo7208/CAD-Coder"
HF_FILE = "cad_data_test_cot.json"
GOLD_CACHE = ROOT / ".cache" / "geoassert" / "cadcoder_gold_all.jsonl"
GOLD_SPLIT_PATH = ROOT / ".cache" / "geoassert" / "gold_split.json"
# Content-addressed reference-program split; about 20% is assigned to test.
_SPLIT_TEST_THRESHOLD = SPLIT_TEST_THRESHOLD

_SYNTAX_REPAIR_TEMPLATE = """\
The CadQuery program below is not valid Python.

## Parse error
{err}

## Current code
```python
{code}
```

Return the corrected, runnable program.
"""


def build_syntax_repair_prompt(code: str, err: str) -> str:
    """Build a repair prompt for source that cannot be parsed."""
    return _SYNTAX_REPAIR_TEMPLATE.format(err=err, code=code.strip())


def gold_partition(gold_code: str) -> str:
    tail = int(hashlib.sha1(gold_code.encode("utf-8")).hexdigest()[-2:], 16)
    return "test" if tail < _SPLIT_TEST_THRESHOLD else "train"


# =====================================================================
# General helpers
# =====================================================================

def _json_safe(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, list):
        return [_json_safe(x) for x in o]
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    return o


@contextlib.contextmanager
def _silence_native_stdout():
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def _strip_fence(code: str) -> str:
    """Remove an optional Markdown code fence."""
    m = re.search(r"```(?:python)?\s*\n?(.*?)```", code, re.DOTALL)
    return (m.group(1) if m else code).strip()


def _n_steps(code: str) -> int:
    try:
        return len(split_steps(code))
    except SyntaxError:
        return 0


def _difficulty(n: int) -> str:
    return "easy" if n <= 2 else ("medium" if n <= 5 else "hard")


def _vlm_template(validity_kind: str, detail: str) -> str:
    return ("The generated part's geometry deviates from the intended design "
            f"(difference: {validity_kind}; {detail}).")


# =====================================================================
# Isolated evaluation worker
# =====================================================================

def _final_ok_shape(report):
    for snap in reversed(report.snapshots):
        if snap.ok and snap.shape is not None:
            return snap.shape
    return None


def _do_evaluate(injected_code: str, gold_code: str, cot: str,
                 n_points: int, step_timeout: float) -> dict:
    try:
        rep = check(injected_code, cot, timeout=step_timeout)
    except SyntaxError as exc:
        # Invalid Python is observably different from the reference program.
        err = f"SyntaxError: {exc}"
        return {"gate2": True, "ok": False, "failing_steps": [],
                "execution_error": None, "exec_payload": err,
                "assert_payload": build_syntax_repair_prompt(injected_code, err),
                "cd": None}
    failing = {r.step.index for r in rep.failures if r.step is not None}
    exec_payload = ""
    if rep.execution_error is not None:
        failing.add(rep.execution_error.step.index)
        if rep.execution_error.error:
            exec_payload = rep.execution_error.error.strip().splitlines()[-1]
    assert_payload = "" if rep.ok else build_repair_prompt(injected_code, rep)

    cd = None
    inj_shape = _final_ok_shape(rep)
    if inj_shape is not None and gold_code:
        try:
            gold_shape = _final_ok_shape(check(gold_code, cot, timeout=step_timeout))
            gpts = None
            if gold_shape is not None:
                ref = gold_reference(gold_shape, n_points,
                                     seed=code_seed(gold_code, n_points))
                gpts, gc, gd = ref["points"], ref["center"], ref["diag"]
            if gpts is not None:
                ipts = sample_points(inj_shape, n_points,
                                     seed=code_seed(injected_code, n_points))
                cd = float(chamfer_distance_points(ipts, gpts, center=gc, diag=gd))
        except Exception:  # noqa: BLE001 - leave the distance unavailable
            cd = None
    return {"gate2": not bool(rep.ok), "ok": bool(rep.ok),
            "failing_steps": sorted(failing),
            "execution_error": (rep.execution_error.step.index
                                if rep.execution_error is not None else None),
            "exec_payload": exec_payload, "assert_payload": assert_payload, "cd": cd}


def _worker_main() -> None:
    import traceback
    raw = sys.stdin.read()
    try:
        with _silence_native_stdout():
            req = json.loads(raw)
            out = _do_evaluate(req["injected_code"], req.get("gold_code", ""),
                               req["cot"], int(req.get("n_points", DEFAULT_N_POINTS)),
                               float(req.get("step_timeout", DEFAULT_STEP_TIMEOUT)))
    except Exception:  # noqa: BLE001
        out = {"worker_error": traceback.format_exc()}
    payload = RESULT_SENTINEL + json.dumps(_json_safe(out), ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def run_eval_worker(injected_code: str, gold_code: str, cot: str, *,
                    n_points: int = DEFAULT_N_POINTS,
                    step_timeout: float = DEFAULT_STEP_TIMEOUT,
                    wall_timeout: float = DEFAULT_WALL_TIMEOUT) -> tuple[dict | None, str]:
    # A subprocess isolates native OCCT crashes and enforces a wall clock.
    req = json.dumps({"injected_code": injected_code, "gold_code": gold_code, "cot": cot,
                      "n_points": n_points, "step_timeout": step_timeout}, ensure_ascii=False)
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        proc = subprocess.run([sys.executable, str(THIS_FILE), "--worker"],
                              input=req.encode("utf-8"), stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=wall_timeout, **kwargs)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if proc.returncode != 0:
        return None, "crash"
    idx = proc.stdout.rfind(RESULT_SENTINEL)
    if idx == -1:
        return None, "crash"
    try:
        data = json.loads(proc.stdout[idx + len(RESULT_SENTINEL):].decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None, "crash"
    if "worker_error" in data:
        return None, "worker_error"
    return data, "ok"


_CRASH_EVAL = {"gate2": True, "ok": False, "failing_steps": [], "execution_error": None,
               "exec_payload": "", "assert_payload": "", "cd": None}


# =====================================================================
# Reference-program loading
# =====================================================================

def pull_golds() -> list:
    if GOLD_CACHE.exists():
        return [json.loads(line) for line in
                GOLD_CACHE.read_text(encoding="utf-8").splitlines() if line.strip()]
    from huggingface_hub import hf_hub_download  # noqa: E402 - scale mode only
    path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset")
    data = json.load(open(path, encoding="utf-8"))
    rows = []
    for item in data:
        msgs = item.get("messages", [])
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        gold = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        if "description:" not in user or not gold:
            continue
        rows.append({"id": str(item.get("model_path", "")).replace(".pth", ""),
                     "gold_code": _strip_fence(gold)})
    GOLD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    GOLD_CACHE.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                          encoding="utf-8")
    return rows


def _gold_seed(gold: str) -> int:
    return int(hashlib.sha1(gold.encode("utf-8")).hexdigest()[:8], 16) & 0x7fffffff


def _make_sample(sid, cls, res, gold, synth, ev, n_steps, diff) -> dict:
    truth_steps = res.mutation.get("truth_steps", [res.injected_step_index])
    # A set-valued label is hit when any labeled step is identified.
    loc_hit = bool(set(truth_steps) & set(ev["failing_steps"]))
    return {
        "id": f"{sid}__{cls}", "source_id": sid, "error_class": cls,
        "injected_code": res.injected_code, "gold_code": gold,
        "injected_step_index": res.injected_step_index,
        "truth_steps": truth_steps,
        "truth_frame": res.mutation.get("truth_frame", "injected"),
        "validity_kind": res.validity_kind, "mutation": res.mutation,
        "cot": synth.cot, "gold_assertions": synth.gold_assertions,
        "n_assertions": synth.n_assertions, "n_steps": n_steps, "difficulty": diff,
        "failing_steps": ev["failing_steps"], "localization_hit": loc_hit,
        "feedback": {"none": "", "exec": ev["exec_payload"], "cd": ev["cd"],
                     "vlm": _vlm_template(res.validity_kind, res.detail),
                     "assert": ev["assert_payload"]},
    }


# =====================================================================
# Dataset generation
# =====================================================================

def build_scale(golds: list, *, target: int, n_points: int, symmetry: bool,
                step_timeout: float, wall_timeout: float, out_path: Path,
                progress: bool = True, tol_pct: float = 2.0,
                ungated_path: Path | None = None, extra_meta: dict | None = None) -> dict:
    # Prioritize programs with Boolean operations for better class coverage.
    golds = sorted(golds, key=lambda g: 0 if any(m in g["gold_code"] for m in BOOL_MARKERS) else 1)

    counters = {c: 0 for c in ERROR_CLASSES}
    seen: set = set()
    reasons: dict = {c: defaultdict(int) for c in ERROR_CLASSES}
    gold_synth: dict = {}
    samples: list = []
    n_gold_processed = 0

    # Resume from existing output and skip completed source/class pairs.
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            s = json.loads(line)
            samples.append(s)
            seen.add((s["source_id"], s["error_class"]))
            counters[s["error_class"]] = counters.get(s["error_class"], 0) + 1
    out_f = open(out_path, "a", encoding="utf-8")
    ungated_f = open(ungated_path, "a", encoding="utf-8") if ungated_path else None

    for gi, g in enumerate(golds, 1):
        if all(counters[c] >= target for c in ERROR_CLASSES):
            break
        gold, sid = g["gold_code"], str(g["id"])
        if all((sid, c) in seen or counters[c] >= target for c in ERROR_CLASSES):
            continue
        key = hashlib.sha1(gold.encode("utf-8")).hexdigest()
        if key not in gold_synth:
            gold_synth[key] = synthesize(gold, symmetry=symmetry,
                                         step_timeout=step_timeout, wall_timeout=wall_timeout,
                                         numeric_tol_pct=tol_pct)
            n_gold_processed += 1
        synth = gold_synth[key]
        if not synth.ok:
            for c in ERROR_CLASSES:
                reasons[c]["gold_unexecutable"] += 1
            continue

        gprops, gstatus = run_gold_props_cached(gold, step_timeout=step_timeout,
                                                wall_timeout=wall_timeout)
        n_steps = _n_steps(gold)
        diff = _difficulty(n_steps)
        for c in ERROR_CLASSES:
            if counters[c] >= target or (sid, c) in seen:
                continue
            res, reason = inject(gold, c, seed=_gold_seed(gold), step_timeout=step_timeout,
                                 wall_timeout=wall_timeout, gold_props=gprops, gold_status=gstatus)
            if res is None:
                reasons[c][reason] += 1
                continue
            ev, status = run_eval_worker(res.injected_code, gold, synth.cot, n_points=n_points,
                                         step_timeout=step_timeout, wall_timeout=wall_timeout)
            if status != "ok":
                ev = dict(_CRASH_EVAL)
            if not ev["gate2"]:
                reasons[c]["injection_not_detected"] += 1
                if ungated_f is not None:
                    ungated_f.write(json.dumps(_json_safe(
                        {**_make_sample(sid, c, res, gold, synth, ev, n_steps, diff),
                         "gate_passed": False}), ensure_ascii=False) + "\n")
                    ungated_f.flush()
                continue
            sample = _make_sample(sid, c, res, gold, synth, ev, n_steps, diff)
            seen.add((sid, c))
            counters[c] += 1
            samples.append(sample)
            out_f.write(json.dumps(_json_safe(sample), ensure_ascii=False) + "\n")
            out_f.flush()
        if progress and gi % 20 == 0:
            print(f"[{gi}/{len(golds)}] synth_golds={n_gold_processed} "
                  + " ".join(f"{c}={counters[c]}" for c in ERROR_CLASSES), flush=True)
    out_f.close()
    if ungated_f is not None:
        ungated_f.close()

    return _build_meta(samples, gold_synth, reasons, counters, target,
                       n_points, symmetry, n_gold_processed, "scale", extra=extra_meta)


# =====================================================================
# Parallel dataset generation; shared counters and file writes stay on the main thread.
# =====================================================================

def _process_gold(g: dict, *, n_points, symmetry, step_timeout, wall_timeout,
                  tol_pct: float = 2.0):
    """Synthesize, mutate, and evaluate one reference program.

    The function does not mutate shared state, so serial and parallel runs use
    the same deterministic sample decisions.
    """
    gold, sid = g["gold_code"], str(g["id"])
    synth = synthesize(gold, symmetry=symmetry, step_timeout=step_timeout,
                       wall_timeout=wall_timeout, numeric_tol_pct=tol_pct)
    if not synth.ok:
        return synth, sid, []
    gprops, gstatus = run_gold_props_cached(gold, step_timeout=step_timeout,
                                            wall_timeout=wall_timeout)
    n_steps = _n_steps(gold)
    diff = _difficulty(n_steps)
    results = []
    for c in ERROR_CLASSES:
        res, reason = inject(gold, c, seed=_gold_seed(gold), step_timeout=step_timeout,
                             wall_timeout=wall_timeout, gold_props=gprops, gold_status=gstatus)
        if res is None:
            results.append((c, reason, None))
            continue
        ev, status = run_eval_worker(res.injected_code, gold, synth.cot, n_points=n_points,
                                     step_timeout=step_timeout, wall_timeout=wall_timeout)
        if status != "ok":
            ev = dict(_CRASH_EVAL)
        if not ev["gate2"]:
            results.append((c, "injection_not_detected",
                            _make_sample(sid, c, res, gold, synth, ev, n_steps, diff)))
            continue
        results.append((c, "ok", _make_sample(sid, c, res, gold, synth, ev, n_steps, diff)))
    return synth, sid, results


def build_scale_parallel(golds: list, *, target: int, jobs: int, n_points: int, symmetry: bool,
                         step_timeout: float, wall_timeout: float, out_path: Path,
                         progress: bool = True, tol_pct: float = 2.0,
                         ungated_path: Path | None = None,
                         extra_meta: dict | None = None) -> dict:
    golds = sorted(golds, key=lambda g: 0 if any(m in g["gold_code"] for m in BOOL_MARKERS) else 1)
    counters = {c: 0 for c in ERROR_CLASSES}
    seen: set = set()
    reasons: dict = {c: defaultdict(int) for c in ERROR_CLASSES}
    samples: list = []
    synth_seen: dict = {}
    n_gold_processed = 0

    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            samples.append(s)
            seen.add((s["source_id"], s["error_class"]))
            counters[s["error_class"]] = counters.get(s["error_class"], 0) + 1
    out_f = open(out_path, "a", encoding="utf-8")
    ungated_f = open(ungated_path, "a", encoding="utf-8") if ungated_path else None

    pending = [g for g in golds
               if not all((str(g["id"]), c) in seen for c in ERROR_CLASSES)]

    def _need(g):
        sid = str(g["id"])
        return not all((sid, c) in seen or counters[c] >= target for c in ERROR_CLASSES)

    def _work(g):
        return _process_gold(g, n_points=n_points, symmetry=symmetry,
                             step_timeout=step_timeout, wall_timeout=wall_timeout,
                             tol_pct=tol_pct)

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        i = 0
        while i < len(pending) and not all(counters[c] >= target for c in ERROR_CLASSES):
            chunk = [g for g in pending[i:i + jobs * 3] if _need(g)]
            i += jobs * 3
            if not chunk:
                continue
            for synth, sid, results in ex.map(_work, chunk):
                n_gold_processed += 1
                synth_seen[sid] = synth
                if not synth.ok:
                    for c in ERROR_CLASSES:
                        reasons[c]["gold_unexecutable"] += 1
                    continue
                for (c, reason, sample) in results:
                    if reason != "ok":
                        reasons[c][reason] += 1
                        if sample is not None and ungated_f is not None:
                            ungated_f.write(json.dumps(_json_safe(
                                {**sample, "gate_passed": False}),
                                ensure_ascii=False) + "\n")
                            ungated_f.flush()
                        continue
                    if (sid, c) in seen:
                        reasons[c]["dedup_collision"] += 1
                        continue
                    if counters[c] >= target:
                        continue
                    seen.add((sid, c))
                    counters[c] += 1
                    samples.append(sample)
                    out_f.write(json.dumps(_json_safe(sample), ensure_ascii=False) + "\n")
                    out_f.flush()
            if progress:
                print(f"[{min(i, len(pending))}/{len(pending)}] golds={n_gold_processed} "
                      + " ".join(f"{c}={counters[c]}" for c in ERROR_CLASSES), flush=True)
    out_f.close()
    if ungated_f is not None:
        ungated_f.close()
    return _build_meta(samples, synth_seen, reasons, counters, target,
                       n_points, symmetry, n_gold_processed, "scale", extra=extra_meta)


# =====================================================================
# Augment an existing gated JSONL file
# =====================================================================

def build_augment(rows: list, *, n_points: int, step_timeout: float, wall_timeout: float,
                  out_path: Path, progress: bool = True) -> dict:
    seen: set = set()
    samples: list = []
    out_f = open(out_path, "w", encoding="utf-8")
    for i, row in enumerate(rows, 1):
        sid = str(row.get("source_id", row.get("id")))
        c = row["error_class"]
        if (sid, c) in seen:
            continue
        ev, status = run_eval_worker(row["injected_code"], row["gold_code"], row["cot"],
                                     n_points=n_points, step_timeout=step_timeout,
                                     wall_timeout=wall_timeout)
        if status != "ok":
            ev = dict(_CRASH_EVAL)
        n_steps = _n_steps(row["gold_code"])
        diff = _difficulty(n_steps)
        loc_hit = row.get("injected_step_index") in ev["failing_steps"]
        sample = {**row, "n_steps": n_steps, "difficulty": diff,
                  "failing_steps": ev["failing_steps"], "localization_hit": loc_hit,
                  "feedback": {"none": "", "exec": ev["exec_payload"], "cd": ev["cd"],
                               "vlm": _vlm_template(row.get("validity_kind", "?"),
                                                    row.get("geo_detail", "")),
                               "assert": ev["assert_payload"]}}
        seen.add((sid, c))
        samples.append(sample)
        out_f.write(json.dumps(_json_safe(sample), ensure_ascii=False) + "\n")
        out_f.flush()
        if progress and i % 20 == 0:
            print(f"[{i}/{len(rows)}] augmented={len(samples)}", flush=True)
    out_f.close()
    counters = defaultdict(int)
    for s in samples:
        counters[s["error_class"]] += 1
    return _build_meta(samples, {}, {c: defaultdict(int) for c in ERROR_CLASSES},
                       counters, None, n_points, False, len(rows), "augment")


# =====================================================================
# 5. meta
# =====================================================================

def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _build_meta(samples, gold_synth, reasons, counters, target, n_points, symmetry,
                n_gold_processed, mode, extra: dict | None = None) -> dict:
    by_class, by_validity, by_diff = defaultdict(int), defaultdict(int), defaultdict(int)
    gold_reuse = defaultdict(int)
    cds, assert_lens = [], []
    n_exec_nonempty = n_loc_hit = n_loc_total = 0
    for s in samples:
        by_class[s["error_class"]] += 1
        by_validity[s.get("validity_kind", "?")] += 1
        by_diff[s.get("difficulty", "?")] += 1
        gold_reuse[s["source_id"]] += 1
        fb = s["feedback"]
        if fb["cd"] is not None:
            cds.append(fb["cd"])
        if fb["exec"]:
            n_exec_nonempty += 1
        assert_lens.append(len(fb["assert"]))
        if s["failing_steps"]:
            n_loc_total += 1
            n_loc_hit += int(s["localization_hit"])
    reuse_hist = defaultdict(int)
    for cnt in gold_reuse.values():
        reuse_hist[cnt] += 1
    golds = list(gold_synth.values())
    return {
        "mode": mode, "n_samples": len(samples), "target_per_class": target,
        "per_class": dict(by_class), "per_validity_kind": dict(by_validity),
        "difficulty_hist": dict(by_diff),
        "under_target": ({c: counters[c] < target for c in ERROR_CLASSES} if target else None),
        "n_golds_processed": n_gold_processed,
        "n_unique_golds_synthesized": len(golds),
        "gold_self_consistent": sum(1 for g in golds if g.ok),
        "gold_reuse_hist": dict(sorted(reuse_hist.items())),
        "inject_reasons": {c: dict(reasons[c]) for c in ERROR_CLASSES},
        "honesty": {
            "harmless_exhausted": {c: reasons[c].get("harmless_exhausted", 0) for c in ERROR_CLASSES},
            "injection_not_detected": {c: reasons[c].get("injection_not_detected", 0) for c in ERROR_CLASSES},
            "localization_hit_rate": (n_loc_hit / n_loc_total) if n_loc_total else None},
        "feedback_stats": {
            "mean_cd": (sum(cds) / len(cds)) if cds else None, "n_with_cd": len(cds),
            "exec_nonempty_frac": (n_exec_nonempty / len(samples)) if samples else None,
            "mean_assert_len": (sum(assert_lens) / len(assert_lens)) if assert_lens else None},
        "config": {"hf_dataset": HF_REPO, "n_points": n_points, "symmetry": symmetry,
                   "error_classes": list(ERROR_CLASSES)},
        **(extra or {}),
    }


def _write_meta(out_path: Path, meta: dict) -> None:
    meta_path = out_path.with_name("benchmark_meta.json")
    meta_path.write_text(json.dumps(_json_safe(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== BENCHMARK META =====")
    print(json.dumps(_json_safe(meta), ensure_ascii=False, indent=2))


# =====================================================================
# Smoke tests
# =====================================================================

def smoke() -> None:
    from _samples import BUGGY_CODE, FIXED_CODE
    print("===== BENCHMARK BUILDER SMOKE TESTS =====")

    assert _difficulty(_n_steps(FIXED_CODE)) == "medium"
    assert _vlm_template("topology", "face_count 7->6")
    print("  [PASS] step count, difficulty, and prompt templates")

    # Evaluate a controlled cutThruAll-to-cutBlind mutation.
    synth = synthesize(FIXED_CODE, symmetry=False)
    assert synth.ok
    ev, status = run_eval_worker(BUGGY_CODE, FIXED_CODE, synth.cot)
    assert status == "ok", f"eval status={status}"
    assert ev["gate2"] is True and 2 in ev["failing_steps"], ev
    assert ev["assert_payload"] and "ASSERTION" in ev["assert_payload"]
    assert isinstance(ev["cd"], float), f"expected numeric Chamfer distance: {ev['cd']}"
    assert ev["exec_payload"] == ""
    print(f"  [PASS] executable mutation: gate2={ev['gate2']} failing={ev['failing_steps']} "
          f"cd={ev['cd']:.4f} assert_len={len(ev['assert_payload'])} exec='{ev['exec_payload']}'")

    # An execution error has feedback but no geometric distance.
    bad = "import cadquery as cq\nresult = cq.Workplane('XY').box(1,1,1).nope()\n"
    ev2, st2 = run_eval_worker(bad, FIXED_CODE, synth.cot)
    assert st2 == "ok" and ev2["gate2"] and ev2["cd"] is None and ev2["exec_payload"], ev2
    print("  [PASS] execution-error payload")

    # Augment a minimal two-row input.
    rows = [{"source_id": "demo", "error_class": "boolean_flip", "injected_code": BUGGY_CODE,
             "gold_code": FIXED_CODE, "cot": synth.cot, "injected_step_index": 2,
             "validity_kind": "topology", "geo_detail": "x"}]
    import tempfile
    tmp = Path(tempfile.gettempdir()) / "t6_smoke_bench.jsonl"
    meta = build_augment(rows, n_points=512, step_timeout=DEFAULT_STEP_TIMEOUT,
                         wall_timeout=DEFAULT_WALL_TIMEOUT, out_path=tmp, progress=False)
    out = [json.loads(l) for l in tmp.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(out) == 1 and set(out[0]["feedback"]) == {"none", "exec", "cd", "vlm", "assert"}
    assert meta["n_samples"] == 1 and "feedback_stats" in meta
    tmp.unlink(missing_ok=True)
    print(f"  [PASS] minimal augmentation (n_samples={meta['n_samples']})")

    print("\nBenchmark builder smoke tests passed.")


# =====================================================================
# Command-line interface
# =====================================================================

if __name__ == "__main__":
    if "--worker" in sys.argv[1:]:
        _worker_main()
        sys.exit(0)

    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Build an experimental CAD mutation benchmark")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mode", choices=["scale", "augment"], default="scale")
    ap.add_argument("--inputs", help="gated JSONL input for augment mode")
    ap.add_argument("--out", help="output benchmark JSONL")
    ap.add_argument("--target", type=int, default=167, help="target samples per class")
    ap.add_argument("--jobs", type=int, default=1, help="parallel reference-program workers")
    ap.add_argument("--n-points", type=int, default=DEFAULT_N_POINTS)
    ap.add_argument("--no-symmetry", action="store_true")
    ap.add_argument("--tol-pct", type=float, default=2.0, help="numeric assertion tolerance in percent")
    ap.add_argument("--split", choices=["test", "train", "all"], default="test",
                    help="reference partition; generated evaluation sets should use test")
    ap.add_argument("--bench-version", default="v2", help="metadata version label")
    ap.add_argument("--no-ungated", action="store_true",
                    help="do not write the undetected-mutation reference set")
    ap.add_argument("--step-timeout", type=float, default=DEFAULT_STEP_TIMEOUT)
    ap.add_argument("--timeout", type=float, default=DEFAULT_WALL_TIMEOUT)
    args = ap.parse_args()
    if args.smoke:
        smoke()
        sys.exit(0)
    if not args.out:
        ap.error("--out is required (or use --smoke)")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "scale":
        golds = pull_golds()
        # Persist the deterministic split for reproducibility and leakage checks.
        for g in golds:
            g["partition"] = gold_partition(g["gold_code"])
        split_counts = {"train": sum(1 for g in golds if g["partition"] == "train"),
                        "test": sum(1 for g in golds if g["partition"] == "test")}
        GOLD_SPLIT_PATH.write_text(json.dumps(
            {"protocol": f"sha1(gold_code)[-2:] < 0x{_SPLIT_TEST_THRESHOLD:02x} → test (≈20%)",
             "counts": split_counts,
             "by_id": {str(g["id"]): g["partition"] for g in golds}},
            ensure_ascii=False), encoding="utf-8")
        if args.split != "all":
            golds = [g for g in golds if g["partition"] == args.split]
        ungated_path = (None if args.no_ungated else
                        out_path.with_name(out_path.stem + "_ungated_ref.jsonl"))
        extra_meta = {"benchmark_version": args.bench_version, "gold_split": args.split,
                      "gold_split_counts": split_counts, "numeric_tol_pct": args.tol_pct,
                      "git_commit": _git_commit(),
                      "ungated_ref": (str(ungated_path.name) if ungated_path else None)}
        print(f"Loaded {len(golds)} reference programs (split={args.split}; all={split_counts}); "
              f"target per class={args.target}; classes={list(ERROR_CLASSES)}; jobs={args.jobs}", flush=True)
        if args.jobs > 1:
            meta = build_scale_parallel(golds, target=args.target, jobs=args.jobs,
                                        n_points=args.n_points, symmetry=not args.no_symmetry,
                                        step_timeout=args.step_timeout, wall_timeout=args.timeout,
                                        out_path=out_path, tol_pct=args.tol_pct,
                                        ungated_path=ungated_path, extra_meta=extra_meta)
        else:
            meta = build_scale(golds, target=args.target, n_points=args.n_points,
                               symmetry=not args.no_symmetry, step_timeout=args.step_timeout,
                               wall_timeout=args.timeout, out_path=out_path,
                               tol_pct=args.tol_pct, ungated_path=ungated_path,
                               extra_meta=extra_meta)
    else:
        if not args.inputs:
            ap.error("augment mode requires --inputs")
        inputs = args.inputs
        rows = [json.loads(line) for line in
                Path(inputs).read_text(encoding="utf-8").splitlines() if line.strip()]
        meta = build_augment(rows, n_points=args.n_points, step_timeout=args.step_timeout,
                             wall_timeout=args.timeout, out_path=out_path)
    _write_meta(out_path, meta)
