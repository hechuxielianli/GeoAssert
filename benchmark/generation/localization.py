"""Compare changed CadQuery steps after AST normalization.

Variable renaming, comments, and formatting are normalized before step sources
are compared. Equal-length programs use positional comparison; programs with
insertions or deletions use a sequence diff. The module reports exact match,
set overlap, precision, and recall against mutation labels.

Examples:
  python localization.py --smoke
  python localization.py --inputs injected.jsonl --out-dir benchmark/generated
"""
from __future__ import annotations

import argparse
import ast
import difflib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from geoassert import split_steps

# Module aliases are never assignment-variable normalization targets.
_PROTECTED_NAMES = frozenset({"cq", "cadquery", "math"})
_UNPARSABLE = {-1}


def _json_safe(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, list):
        return [_json_safe(x) for x in o]
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    return o


# =====================================================================
# AST normalization
# =====================================================================

def normalize_code(code: str) -> str | None:
    """Normalize assignment names and formatting in a CadQuery program.

    Assignment targets are renamed by first appearance. Module aliases,
    attribute names, and selector strings are preserved. Invalid Python
    returns ``None``.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    mapping: dict[str, str] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                and node.id not in _PROTECTED_NAMES and node.id not in mapping):
            mapping[node.id] = f"_v{len(mapping)}"

    class _Renamer(ast.NodeTransformer):
        def visit_Name(self, n: ast.Name):
            if n.id in mapping:
                return ast.copy_location(ast.Name(id=mapping[n.id], ctx=n.ctx), n)
            return n

    new = _Renamer().visit(tree)
    ast.fix_missing_locations(new)
    return ast.unparse(new)


def _sources(norm_code: str) -> list[str] | None:
    """Return normalized source for each step, or ``None`` if parsing fails."""
    try:
        return [s.step_source for s in split_steps(norm_code)]
    except SyntaxError:
        return None


# =====================================================================
# Changed-step detection
# =====================================================================

def changed_steps(code_a: str, code_b: str, *, normalize: bool = True,
                  collapse_runs: bool = True) -> set[int]:
    """Return one-based steps in ``code_b`` that differ from ``code_a``.

    Equal-length step sequences use positional comparison so swaps identify
    both positions. Unequal sequences use a sequence diff. With
    ``collapse_runs=True``, each contiguous changed region is represented by
    its first step. Invalid ``code_b`` returns the sentinel ``{-1}``.
    """
    na = normalize_code(code_a) if normalize else code_a
    nb = normalize_code(code_b) if normalize else code_b
    if nb is None:
        return set(_UNPARSABLE)
    if na is not None and na == nb:
        return set()
    sb = _sources(nb)
    if sb is None:
        return set(_UNPARSABLE)
    sa = _sources(na) if na is not None else None
    if sa is None:
        return set(range(1, len(sb) + 1))

    if len(sa) == len(sb):
        changed = {j + 1 for j, (x, y) in enumerate(zip(sa, sb)) if x != y}
        if collapse_runs:
            changed = {p for p in changed if p - 1 not in changed}
        return changed

    changed = set()
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(None, sa, sb,
                                                         autojunk=False).get_opcodes():
        if tag in ("replace", "insert"):
            changed.update([j1 + 1] if collapse_runs else range(j1 + 1, j2 + 1))
    return changed


# =====================================================================
# Metrics
# =====================================================================

def loc_metrics(predicted: set[int], truth: set[int]) -> dict:
    """Compute exact match, coverage, precision, and recall for one sample."""
    inter = predicted & truth
    return {
        "exact": predicted == truth,
        "hit": truth <= predicted,
        "precision": (len(inter) / len(predicted)) if predicted else 0.0,
        "recall": (len(inter) / len(truth)) if truth else 1.0,
    }


def assertion_localized_steps(report) -> set[int]:
    """Return steps identified by failed assertions or an execution error."""
    steps = {r.step.index for r in report.failures if r.step is not None}
    err = getattr(report, "execution_error", None)
    if err is not None and getattr(err, "step", None) is not None:
        steps.add(err.step.index)
    return steps


# =====================================================================
# Batch evaluation
# =====================================================================

def evaluate(rows: list, *, collapse_runs: bool = True) -> dict:
    by_class: dict[str, list] = defaultdict(list)
    failures: list = []
    n_syntax_skipped = 0
    for row in rows:
        gold, inj = row.get("gold_code"), row.get("injected_code")
        truth_idx = row.get("injected_step_index")
        if not gold or not inj or truth_idx is None:
            continue
        if row.get("error_class") == "syntax_error":
            n_syntax_skipped += 1
            continue
        cls = row.get("error_class", "?")
        pred = changed_steps(gold, inj, collapse_runs=collapse_runs)
        ts = row.get("truth_steps")
        truth = {int(x) for x in ts} if ts else {int(truth_idx)}
        m = loc_metrics(pred, truth)
        by_class[cls].append(m)
        if not m["exact"]:
            failures.append({"id": row.get("id"), "error_class": cls,
                             "validity_kind": row.get("validity_kind"),
                             "truth": sorted(truth), "predicted": sorted(pred)})

    def agg(ms: list) -> dict:
        n = len(ms)
        if n == 0:
            return {"n": 0}
        return {"n": n,
                "exact": sum(m["exact"] for m in ms) / n,
                "hit": sum(m["hit"] for m in ms) / n,
                "precision": sum(m["precision"] for m in ms) / n,
                "recall": sum(m["recall"] for m in ms) / n}

    per_class = {c: agg(ms) for c, ms in sorted(by_class.items())}
    overall = agg([m for ms in by_class.values() for m in ms])
    return {"per_class": per_class, "overall": overall, "failures": failures,
            "n_syntax_skipped": n_syntax_skipped}


def main(inputs: str, out_dir: str, n: int) -> None:
    rows = [json.loads(line) for line in
            Path(inputs).read_text(encoding="utf-8").splitlines() if line.strip()]
    if n > 0:
        rows = rows[:n]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    first = evaluate(rows, collapse_runs=True)
    full = evaluate(rows, collapse_runs=False)
    result = {
        "n_rows": len(rows),
        "truth": "injected_step_index (causal mutation step)",
        "first_only": {"per_class": first["per_class"], "overall": first["overall"]},
        "full_run": {"per_class": full["per_class"], "overall": full["overall"]},
        "failures_full_run": full["failures"],
        "failures_first_only": first["failures"],
    }
    (out / "localization_metrics.json").write_text(
        json.dumps(_json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== LOCALIZATION METRICS =====")
    for name, ev in (("FIRST-only", first), ("FULL-run", full)):
        o = ev["overall"]
        print(f"[{name:10s}] overall  exact={o['exact']:.3f} hit={o['hit']:.3f} "
              f"precision={o['precision']:.3f} recall={o['recall']:.3f}  (n={o['n']})")
        for c, m in ev["per_class"].items():
            print(f"             {c:14s} exact={m['exact']:.3f} hit={m['hit']:.3f} "
                  f"precision={m['precision']:.3f} (n={m['n']})")
    print(f"\nFULL-run non-exact samples: {len(full['failures'])}; "
          f"FIRST-only non-exact samples: {len(first['failures'])}")


# =====================================================================
# Offline smoke tests
# =====================================================================

_BOX = ("import cadquery as cq\n"
        "result = cq.Workplane('XY').box(100, 60, 40)\n"
        "result = result.faces('>Z').workplane().circle(10).cutBlind(5)\n"
        "result = result.edges('|Z').fillet(5)\n")
_MANY = ("import cadquery as cq\n"
         "result = cq.Workplane('XY').box(40, 40, 40)\n"
         "result = result.faces('>Z').circle(10).cutThruAll().faces('<Z').hole(4)\n")


def smoke() -> None:
    print("===== LOCALIZATION SMOKE TESTS =====")

    assert changed_steps(_BOX, _BOX.replace("result", "r")) == set(), "renaming changed the result"
    commented = _BOX.replace("box(100, 60, 40)\n", "box(100, 60, 40)  # base\n")
    assert changed_steps(_BOX, commented) == set(), "comments changed the result"
    assert changed_steps(_BOX, _BOX.replace("'XY'", '"XY"').replace("'>Z'", '">Z"')) == set(), \
        "quote style changed the result"
    print("  [PASS] renaming, comments, and quote style normalize identically")

    assert changed_steps(_BOX, _BOX.replace("cutBlind(5)", "cutBlind(-5)")) == {2}
    fixed_code = _BOX.replace("cutBlind(5)", "cutThruAll()")
    assert changed_steps(_BOX, fixed_code) == {2}
    print("  [PASS] single-step mutations localize to step 2")

    assert changed_steps(_BOX, "not python (") == {-1}
    print("  [PASS] invalid Python returns the {-1} sentinel")

    m2o = _MANY.replace("circle(10)", "circle(8)")
    full = changed_steps(_MANY, m2o, collapse_runs=False)
    first = changed_steps(_MANY, m2o, collapse_runs=True)
    assert full == {2, 3} and first == {2}, f"full={full} first={first}"
    print("  [PASS] contiguous changes collapse to their first step")

    m = loc_metrics({2, 3}, {2})
    assert m["exact"] is False and m["hit"] is True
    assert abs(m["precision"] - 0.5) < 1e-9 and abs(m["recall"] - 1.0) < 1e-9
    assert loc_metrics(set(), {2})["precision"] == 0.0
    print("  [PASS] localization metrics")

    # Operation omission begins at the removed position.
    m2o_drop = _MANY.replace(".cutThruAll()", "")
    assert changed_steps(_MANY, m2o_drop) == {2}, changed_steps(_MANY, m2o_drop)
    # A trailing deletion has no corresponding position in code_b.
    tail_drop = "\n".join(_BOX.splitlines()[:-1]) + "\n"
    assert changed_steps(_BOX, tail_drop) == set()
    # Reordering identifies both swapped positions.
    ls = _BOX.splitlines()
    swapped = "\n".join([ls[0], ls[1], ls[3], ls[2]]) + "\n"
    assert changed_steps(_BOX, swapped, collapse_runs=False) == {2, 3}
    assert loc_metrics({2, 3}, {2, 3})["exact"] is True
    print("  [PASS] omission and reordering conventions")

    # Validate an optional generated sample set when present.
    sp = ROOT / "benchmark" / "samples" / "injected_samples.jsonl"
    if sp.exists():
        rows = [json.loads(l) for l in sp.read_text(encoding="utf-8").splitlines() if l.strip()]
        fo = evaluate(rows, collapse_runs=True)["overall"]
        fu = evaluate(rows, collapse_runs=False)["overall"]
        print(f"  [PASS] generated samples (n={fo['n']}): FIRST exact={fo['exact']:.3f} hit={fo['hit']:.3f} "
              f"| FULL exact={fu['exact']:.3f} hit={fu['hit']:.3f}")
        assert fo["exact"] == 1.0 and fo["hit"] == 1.0, f"unexpected FIRST-only metrics: {fo}"
        assert fu["hit"] == 1.0, f"unexpected FULL-run metrics: {fu}"
    else:
        print("  [SKIP] generated sample set is not present")

    print("\nLocalization smoke tests passed.")


# =====================================================================
# Command-line interface
# =====================================================================

if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Evaluate changed-step localization")
    ap.add_argument("--smoke", action="store_true", help="run offline AST smoke tests")
    ap.add_argument("--inputs")
    ap.add_argument("--out-dir")
    ap.add_argument("--n", type=int, default=0, help="maximum rows; 0 means all")
    args = ap.parse_args()

    if args.smoke:
        smoke()
    else:
        if not args.inputs or not args.out_dir:
            ap.error("batch mode requires --inputs and --out-dir (or use --smoke)")
        main(inputs=args.inputs, out_dir=args.out_dir, n=args.n)
