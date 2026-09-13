"""Filter mutations using assertions synthesized from reference programs.

A sample passes when the reference program is executable and self-consistent,
and the mutated program either violates at least one reference assertion or
fails during execution. Undetected but geometrically effective mutations are
counted separately rather than silently discarded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATION_DIR = Path(__file__).resolve().parent
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))
from synthesize_assertions import (  # noqa: E402
    DEFAULT_STEP_TIMEOUT, DEFAULT_WALL_TIMEOUT, SynthResult, _json_safe,
    run_check_worker, synthesize,
)


@dataclass
class GateOutcome:
    passed: bool
    gate1_passed: bool
    gate2_passed: bool
    injected_check_ok: bool | None
    injected_check_status: str
    failing_steps: list
    localization_hit: bool
    reason: str | None


def gate_sample(inj_row: dict, synth: SynthResult, *,
                step_timeout: float = DEFAULT_STEP_TIMEOUT,
                wall_timeout: float = DEFAULT_WALL_TIMEOUT) -> GateOutcome:
    if not synth.ok:
        return GateOutcome(False, False, False, None, "skipped", [], False, synth.drop_reason)

    rep, status = run_check_worker(inj_row["injected_code"], synth.cot,
                                   timeout=step_timeout, wall_timeout=wall_timeout)
    if status != "ok":
        # A timeout or crash is treated as an observable mutation failure.
        return GateOutcome(True, True, True, False, status, [], False, None)

    injected_ok = bool(rep["ok"])
    gate2 = injected_ok is False
    failing = {f["step"] for f in rep["failures"] if f["step"] is not None}
    if rep.get("execution_error") is not None:
        failing.add(rep["execution_error"])
    failing_steps = sorted(failing)
    loc_hit = inj_row.get("injected_step_index") in failing_steps
    return GateOutcome(gate2, True, gate2, injected_ok, "ok", failing_steps, loc_hit,
                       None if gate2 else "injection_not_detected")


def build_gated_dataset(rows: list, *, symmetry: bool = True,
                        step_timeout: float = DEFAULT_STEP_TIMEOUT,
                        wall_timeout: float = DEFAULT_WALL_TIMEOUT,
                        progress: bool = False) -> tuple[list, dict]:
    synth_cache: dict[str, SynthResult] = {}

    def get_synth(gold: str) -> SynthResult:
        key = hashlib.sha1(gold.encode("utf-8")).hexdigest()
        if key not in synth_cache:
            synth_cache[key] = synthesize(gold, symmetry=symmetry,
                                          step_timeout=step_timeout, wall_timeout=wall_timeout)
        return synth_cache[key]

    out_rows: list = []
    per_class = defaultdict(lambda: {"n": 0, "gate2_pass": 0})
    per_validity = defaultdict(lambda: {"n": 0, "gate2_pass": 0})
    drop_reasons: dict[str, int] = defaultdict(int)
    n_loc_hit = n_loc_total = 0

    for i, row in enumerate(rows, 1):
        gold = row.get("gold_code")
        if not gold:
            continue
        synth = get_synth(gold)
        outcome = gate_sample(row, synth, step_timeout=step_timeout, wall_timeout=wall_timeout)
        cls, vk = row.get("error_class", "?"), row.get("validity_kind", "?")
        per_class[cls]["n"] += 1
        per_validity[vk]["n"] += 1
        if outcome.gate2_passed:
            per_class[cls]["gate2_pass"] += 1
            per_validity[vk]["gate2_pass"] += 1
        if outcome.reason:
            drop_reasons[outcome.reason] += 1
        if outcome.passed:
            if outcome.failing_steps:
                n_loc_total += 1
                n_loc_hit += int(outcome.localization_hit)
            out_rows.append({**row,
                             "cot": synth.cot, "gold_assertions": synth.gold_assertions,
                             "gold_check_ok": synth.ok,
                             "injected_check_ok": outcome.injected_check_ok,
                             "injected_check_status": outcome.injected_check_status,
                             "gate1_passed": outcome.gate1_passed,
                             "gate2_passed": outcome.gate2_passed, "gate_passed": True,
                             "failing_steps": outcome.failing_steps,
                             "localization_hit": outcome.localization_hit,
                             "n_assertions": synth.n_assertions})
        if progress:
            print(f"[{i}/{len(rows)}] {row.get('id','?'):28s} "
                  f"gate1={outcome.gate1_passed} gate2={outcome.gate2_passed} "
                  f"reason={outcome.reason or '-'}")

    golds = list(synth_cache.values())
    gold_ok = sum(1 for s in golds if s.ok)
    gold_drop = defaultdict(int)
    for s in golds:
        if not s.ok:
            gold_drop[s.drop_reason or "unknown"] += 1
    synth_ok = [s for s in golds if s.ok]
    meta = {
        "n_unique_golds": len(golds),
        "gold_self_consistent": gold_ok,
        "gold_self_consistent_rate": (gold_ok / len(golds)) if golds else None,
        "gold_drop_reasons": dict(gold_drop),
        "n_injected_samples": sum(p["n"] for p in per_class.values()),
        "gate2_passed": sum(p["gate2_pass"] for p in per_class.values()),
        "gate2_rate": (sum(p["gate2_pass"] for p in per_class.values())
                       / sum(p["n"] for p in per_class.values())) if per_class else None,
        "n_gated_in": len(out_rows),
        "per_class": {c: {**v, "rate": (v["gate2_pass"] / v["n"]) if v["n"] else None}
                      for c, v in sorted(per_class.items())},
        "per_validity_kind": {k: {**v, "rate": (v["gate2_pass"] / v["n"]) if v["n"] else None}
                              for k, v in sorted(per_validity.items())},
        "drop_reasons": dict(drop_reasons),
        "injection_not_detected": drop_reasons.get("injection_not_detected", 0),
        "localization_hit_rate": (n_loc_hit / n_loc_total) if n_loc_total else None,
        "assertion_stats": {
            "mean_per_gold": (sum(s.n_assertions for s in synth_ok) / len(synth_ok))
                             if synth_ok else None,
            "n_dropped_total": sum(s.n_dropped for s in synth_ok),
            "golds_with_symmetry": sum(1 for s in synth_ok if s.has_symmetry)},
        "symmetry": {"enabled": symmetry,
                     "sym_status_counts": dict(_count(s.sym_status for s in golds))},
        "config": {"numeric_tol_pct": 2, "sym_planes": ["XY", "XZ", "YZ"],
                   "step_timeout": step_timeout, "wall_timeout": wall_timeout},
    }
    return out_rows, meta


def _count(it):
    c = defaultdict(int)
    for x in it:
        c[x] += 1
    return c


def main(inputs: str, out_dir: str, n: int, symmetry: bool,
         step_timeout: float, wall_timeout: float) -> None:
    rows = [json.loads(line) for line in
            Path(inputs).read_text(encoding="utf-8").splitlines() if line.strip()]
    if n > 0:
        rows = rows[:n]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    out_rows, meta = build_gated_dataset(rows, symmetry=symmetry, step_timeout=step_timeout,
                                         wall_timeout=wall_timeout, progress=True)
    (out / "benchmark_gated.jsonl").write_text(
        "\n".join(json.dumps(_json_safe(r), ensure_ascii=False) for r in out_rows) + "\n",
        encoding="utf-8")
    (out / "t5_gate_meta.json").write_text(
        json.dumps(_json_safe(meta), ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== GATE METADATA =====")
    print(json.dumps(_json_safe(meta), ensure_ascii=False, indent=2))


# =====================================================================
# Smoke tests
# =====================================================================

def smoke(*, symmetry: bool = True) -> None:
    from _samples import BUGGY_CODE, FIXED_CODE
    print("===== GATE SMOKE TESTS =====")

    synth = synthesize(FIXED_CODE, symmetry=symmetry)
    assert synth.ok, f"gold synth not ok: {synth.drop_reason}"

    inj = {"injected_code": BUGGY_CODE, "injected_step_index": 2,
           "error_class": "boolean_flip", "validity_kind": "topology", "id": "demo__inj"}
    oc = gate_sample(inj, synth)
    assert oc.gate2_passed and 2 in oc.failing_steps and oc.localization_hit, \
        f"mutation was not detected at step 2: {oc}"
    print(f"  [PASS] mutation detected: failing_steps={oc.failing_steps} loc_hit={oc.localization_hit}")

    noop = {"injected_code": FIXED_CODE, "injected_step_index": 2,
            "error_class": "x", "validity_kind": "topology", "id": "noop"}
    oc2 = gate_sample(noop, synth)
    assert not oc2.gate2_passed and oc2.reason == "injection_not_detected", \
        f"no-op mutation unexpectedly passed: {oc2}"
    print("  [PASS] no-op mutation rejected")

    bad = {"injected_code": "import cadquery as cq\nresult = cq.Workplane('XY').box(1,1,1).nonexistent_op()\n",
           "injected_step_index": 1, "error_class": "api_misuse", "validity_kind": "exec_error", "id": "exec"}
    oc3 = gate_sample(bad, synth)
    assert oc3.gate2_passed and oc3.injected_check_ok is False, f"execution error was not detected: {oc3}"
    print("  [PASS] execution error detected")

    rows = [{**inj, "gold_code": FIXED_CODE}, {**bad, "gold_code": FIXED_CODE}]
    out_rows, meta = build_gated_dataset(rows, symmetry=False)
    assert meta["n_unique_golds"] == 1 and meta["gate2_rate"] is not None
    assert len(out_rows) == 2
    print(f"  [PASS] two samples gated; detection rate={meta['gate2_rate']:.2f}; "
          f"reference consistency={meta['gold_self_consistent_rate']:.2f}")

    print("\nGate smoke tests passed.")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Filter mutations with executable assertions")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--inputs")
    ap.add_argument("--out-dir")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--no-symmetry", action="store_true", help="skip symmetry checks")
    ap.add_argument("--step-timeout", type=float, default=DEFAULT_STEP_TIMEOUT)
    ap.add_argument("--timeout", type=float, default=DEFAULT_WALL_TIMEOUT)
    args = ap.parse_args()
    if args.smoke:
        smoke(symmetry=not args.no_symmetry)
    else:
        if not args.inputs or not args.out_dir:
            ap.error("batch mode requires --inputs and --out-dir (or use --smoke)")
        main(inputs=args.inputs, out_dir=args.out_dir, n=args.n,
             symmetry=not args.no_symmetry, step_timeout=args.step_timeout,
             wall_timeout=args.timeout)
