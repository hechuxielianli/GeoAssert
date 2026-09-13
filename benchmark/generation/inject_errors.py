"""Generate controlled CadQuery program mutations with step-level labels.

The generator supports API substitutions, numeric parameter shifts, boolean
reversals, syntax errors, operation omissions, and operation reordering.
Candidate programs execute in subprocesses so native OCCT failures do not stop
the batch. Mutations that execute but do not change measured geometry are
rejected and counted in the generated metadata.

Examples:
  python inject_errors.py --smoke
  python inject_errors.py --inputs gold.jsonl --out-dir benchmark/generated
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THIS_FILE = Path(__file__).resolve()
from geoassert.checker import (
    GeoProps, TERMINAL_OPS, _CONTROL_FLOW, _chain_calls, _chain_root_ok,
    execute_steps, split_steps,
)
from geoassert.constants import GEO_DIFF_REL_TOL

RESULT_SENTINEL = b"@@INJECT_RESULT@@"
DEFAULT_STEP_TIMEOUT = 30.0
DEFAULT_WALL_TIMEOUT = 120.0
REL_TOL = GEO_DIFF_REL_TOL
# Labels use injected-program steps, gold positions, or a set of swapped steps.
ERROR_CLASSES = ("api_misuse", "param_shift", "boolean_flip",
                 "syntax_error", "operation_omission", "reorder")

# API substitutions retain a compatible call signature where possible.
API_MISUSE_PURE = {"fillet": "chamfer", "chamfer": "fillet", "extrude": "cutBlind"}
API_MISUSE_COMPOUND = {"cutThruAll": ("cutBlind", "add_arg"),
                       "cutBlind": ("cutThruAll", "drop_arg")}
BOOL_FLIP = {"cut": "union", "union": "cut", "intersect": "union"}
SELECTOR_FLIP = {">Z": "<Z", "<Z": ">Z", ">X": "<X", "<X": ">X", ">Y": "<Y", "<Y": ">Y"}
SHIFT_MODES = ("negate", "scale_down", "scale_up", "offset")


# =====================================================================
# General helpers
# =====================================================================

def _parses(code: str) -> bool:
    if not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _json_safe(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, list):
        return [_json_safe(x) for x in o]
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    return o


# =====================================================================
# Step enumeration, aligned with geoassert.split_steps
# =====================================================================

@dataclass(frozen=True)
class TerminalSite:
    step_index: int
    op_name: str
    stmt_index: int
    chain_pos: int


def _has_control_flow(tree: ast.Module) -> bool:
    return any(isinstance(s, _CONTROL_FLOW) for s in tree.body)


def _chain_statements(tree: ast.Module):
    """Yield statements, calls, terminal positions, and one-based steps."""
    known_vars: set[str] = set()
    step = 0
    for si, stmt in enumerate(tree.body):
        is_chain = (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and _chain_root_ok(stmt.value, known_vars)
            and _chain_calls(stmt.value) is not None
        )
        if not is_chain:
            continue
        known_vars.add(stmt.targets[0].id)
        calls = _chain_calls(stmt.value)
        if not calls:
            continue
        term_idx = [i for i, (n, _) in enumerate(calls) if n in TERMINAL_OPS]
        if not term_idx:
            term_idx = [len(calls) - 1]
        term_idx[-1] = len(calls) - 1
        term_steps = [step + 1 + j for j in range(len(term_idx))]
        step += len(term_idx)
        yield si, stmt, calls, term_idx, term_steps


def _enumerate_terminals(tree: ast.Module) -> list[TerminalSite]:
    sites: list[TerminalSite] = []
    for si, _stmt, calls, tpos, tstep in _chain_statements(tree):
        for k, ts in zip(tpos, tstep):
            sites.append(TerminalSite(ts, calls[k][0], si, k))
    return sites


def _owner_step(pos: int, tpos: list[int], tstep: list[int]) -> int:
    """Return the terminal step that owns a position in a call chain."""
    for tp, ts in zip(tpos, tstep):
        if pos <= tp:
            return ts
    return tstep[-1]


def assert_step_consistency(code: str) -> None:
    """Check that mutation labels use the checker's step convention."""
    tree = ast.parse(code)
    if _has_control_flow(tree):
        return
    mine = [s.op_name for s in _enumerate_terminals(tree)]
    if not mine:
        return
    theirs = [s.label for s in split_steps(code)]
    if mine != theirs:
        raise AssertionError(f"step enumeration mismatch: {mine} != {theirs}\n{code}")


# =====================================================================
# Deterministic AST mutation candidates
# =====================================================================

def _num_value(node) -> float | None:
    """Extract a numeric literal, including a unary negative literal."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return float(node.value)
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
            and not isinstance(node.operand.value, bool)):
        return -float(node.operand.value)
    return None


def _make_num_node(v: float):
    iv = int(v) if float(v).is_integer() else v
    if iv < 0:
        return ast.UnaryOp(op=ast.USub(), operand=ast.Constant(abs(iv)))
    return ast.Constant(iv)


def _numeric_sites_in_call(call: ast.Call) -> list:
    """Return mutable numeric argument sites, including tuple/list members."""
    sites: list = []

    def scan(container: list) -> None:
        for i, elt in enumerate(container):
            val = _num_value(elt)
            if val is not None:
                def setter(new, _c=container, _i=i):
                    _c[_i] = new
                sites.append((val, setter))
            elif isinstance(elt, (ast.Tuple, ast.List)):
                scan(elt.elts)

    scan(call.args)
    return sites


def _apply_shift(val: float, mode: str) -> float:
    return {"negate": -val, "scale_down": val * 0.1,
            "scale_up": val * 2.0, "offset": val + 5.0}[mode]


def _candidates_param_shift(tree: ast.Module) -> list:
    sites = []   # (si, k, ci, owner, value)
    for si, _stmt, calls, tpos, tstep in _chain_statements(tree):
        for k, (_name, call) in enumerate(calls):
            for ci, (val, _setter) in enumerate(_numeric_sites_in_call(call)):
                sites.append((si, k, ci, _owner_step(k, tpos, tstep), val))
    out = []
    for (si, k, ci, owner, val) in sites:
        for mode in SHIFT_MODES:
            new_val = _apply_shift(val, mode)
            if new_val == val:
                continue
            cp = copy.deepcopy(tree)
            call_cp = _chain_calls(cp.body[si].value)[k][1]
            _val_cp, setter = _numeric_sites_in_call(call_cp)[ci]
            setter(_make_num_node(new_val))
            ast.fix_missing_locations(cp)
            code = ast.unparse(cp)
            if _parses(code):
                out.append((code, owner, {"class": "param_shift", "mode": mode,
                                          "from": val, "to": new_val, "op_pos": k}))
            break
    return out


def _candidates_api_misuse(tree: ast.Module) -> list:
    sites = []   # (si, k, owner, name)
    for si, _stmt, calls, tpos, tstep in _chain_statements(tree):
        for k in tpos:
            name = calls[k][0]
            if name in API_MISUSE_PURE or name in API_MISUSE_COMPOUND:
                sites.append((si, k, _owner_step(k, tpos, tstep), name))
    out = []
    for (si, k, owner, name) in sites:
        cp = copy.deepcopy(tree)
        node = _chain_calls(cp.body[si].value)[k][1]
        if name in API_MISUSE_PURE:
            to = API_MISUSE_PURE[name]
            node.func.attr = to
            detail = {"sub": "rename", "from": name, "to": to}
        else:
            to, how = API_MISUSE_COMPOUND[name]
            node.func.attr = to
            if how == "drop_arg":
                node.args, node.keywords = [], []
            else:
                node.args, node.keywords = [ast.Constant(1.0)], []
            detail = {"sub": "compound", "from": name, "to": to, "how": how}
        ast.fix_missing_locations(cp)
        code = ast.unparse(cp)
        if _parses(code):
            out.append((code, owner, {"class": "api_misuse", **detail}))
    return out


def _candidates_boolean_flip(tree: ast.Module) -> list:
    bsites, ssites = [], []
    for si, _stmt, calls, tpos, tstep in _chain_statements(tree):
        for k, (name, call) in enumerate(calls):
            if name in BOOL_FLIP:
                bsites.append((si, k, _owner_step(k, tpos, tstep), name))
            for ai, arg in enumerate(call.args):
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                        and arg.value in SELECTOR_FLIP:
                    ssites.append((si, k, ai, _owner_step(k, tpos, tstep), arg.value))
    out = []
    for (si, k, owner, name) in bsites:
        cp = copy.deepcopy(tree)
        node = _chain_calls(cp.body[si].value)[k][1]
        node.func.attr = BOOL_FLIP[name]
        ast.fix_missing_locations(cp)
        code = ast.unparse(cp)
        if _parses(code):
            out.append((code, owner, {"class": "boolean_flip", "sub": "method",
                                      "from": name, "to": BOOL_FLIP[name]}))
    for (si, k, ai, owner, sel) in ssites:
        cp = copy.deepcopy(tree)
        call = _chain_calls(cp.body[si].value)[k][1]
        call.args[ai] = ast.Constant(SELECTOR_FLIP[sel])
        ast.fix_missing_locations(cp)
        code = ast.unparse(cp)
        if _parses(code):
            out.append((code, owner, {"class": "boolean_flip", "sub": "selector",
                                      "from": sel, "to": SELECTOR_FLIP[sel]}))
    return out


# Syntax errors are introduced at the text level.

_SYNTAX_MODES = ("drop_close_paren", "drop_open_paren")


def _candidates_syntax(tree: ast.Module) -> list:
    """Remove one parenthesis and label the affected reference position."""
    canonical = ast.unparse(tree)
    lines = canonical.splitlines()
    if len(lines) != len(tree.body):
        return []
    out = []
    for si, _stmt, _calls, _tpos, tstep in _chain_statements(tree):
        line = lines[si]
        for mode in _SYNTAX_MODES:
            idx = line.rfind(")") if mode == "drop_close_paren" else line.find("(")
            if idx == -1:
                continue
            code = "\n".join([*lines[:si], line[:idx] + line[idx + 1:], *lines[si + 1:]])
            if _parses(code):
                continue
            owner = tstep[0]
            out.append((code, owner, {"class": "syntax_error", "mode": mode,
                                      "line": si, "gold_step": owner,
                                      "truth_frame": "gold_position",
                                      "truth_steps": [owner]}))
            break
    return out


# Operation omission

def _candidates_omission(tree: ast.Module) -> list:
    """Remove an operation and label the first divergent injected step."""
    out = []
    # Remove one terminal call from a chain.
    for si, _stmt, calls, tpos, tstep in _chain_statements(tree):
        for k in tpos:
            name, call = calls[k]
            if name not in TERMINAL_OPS or not isinstance(call.func, ast.Attribute):
                continue
            cp = copy.deepcopy(tree)
            cp_calls = _chain_calls(cp.body[si].value)
            inner = cp_calls[k][1].func.value
            if k == len(cp_calls) - 1:
                cp.body[si].value = inner
            else:
                cp_calls[k + 1][1].func.value = inner
            ast.fix_missing_locations(cp)
            code = ast.unparse(cp)
            if not _parses(code):
                continue
            gold_step = _owner_step(k, tpos, tstep)
            n_inj = len(_enumerate_terminals(ast.parse(code))) or 1
            owner = min(gold_step, n_inj)
            out.append((code, owner, {"class": "operation_omission", "sub": "drop_op",
                                      "removed": name, "gold_step": gold_step,
                                      "truth_frame": "gold_position",
                                      "truth_steps": [owner]}))
    # Remove a reassignment whose target was already defined.
    first_def: dict[str, int] = {}
    for i, s in enumerate(tree.body):
        if isinstance(s, ast.Assign) and len(s.targets) == 1 \
                and isinstance(s.targets[0], ast.Name):
            first_def.setdefault(s.targets[0].id, i)
    for si, stmt, _calls, _tpos, tstep in _chain_statements(tree):
        if first_def.get(stmt.targets[0].id) == si:
            continue
        cp = copy.deepcopy(tree)
        del cp.body[si]
        code = ast.unparse(cp)
        if not _parses(code):
            continue
        gold_step = tstep[0]
        n_inj = len(_enumerate_terminals(ast.parse(code))) or 1
        owner = min(gold_step, n_inj)
        out.append((code, owner, {"class": "operation_omission", "sub": "drop_stmt",
                                  "removed": f"stmt#{si}", "gold_step": gold_step,
                                  "truth_frame": "gold_position",
                                  "truth_steps": [owner]}))
    return out


# Reorder adjacent chain statements

def _stmt_loads(stmt) -> set:
    return {n.id for n in ast.walk(stmt)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _candidates_reorder(tree: ast.Module) -> list:
    """Swap adjacent chains and label every swapped terminal step."""
    rows = list(_chain_statements(tree))
    first_def: dict[str, int] = {}
    for i, s in enumerate(tree.body):
        if isinstance(s, ast.Assign) and len(s.targets) == 1 \
                and isinstance(s.targets[0], ast.Name):
            first_def.setdefault(s.targets[0].id, i)
    out = []
    for (a_si, a_stmt, _ac, _ap, a_steps), (b_si, b_stmt, _bc, _bp, b_steps) \
            in zip(rows, rows[1:]):
        if b_si != a_si + 1:
            continue
        tgt_a = a_stmt.targets[0].id
        # Reject swaps that would load a name before its first definition.
        if first_def.get(tgt_a) == a_si and tgt_a in _stmt_loads(b_stmt):
            continue
        cp = copy.deepcopy(tree)
        cp.body[a_si], cp.body[b_si] = cp.body[b_si], cp.body[a_si]
        code = ast.unparse(cp)
        if not _parses(code):
            continue
        steps = sorted(set(a_steps) | set(b_steps))
        out.append((code, steps[0], {"class": "reorder", "swapped": [a_si, b_si],
                                     "truth_frame": "set", "truth_steps": steps}))
    return out


CANDIDATE_FN = {"api_misuse": _candidates_api_misuse,
                "param_shift": _candidates_param_shift,
                "boolean_flip": _candidates_boolean_flip,
                "syntax_error": _candidates_syntax,
                "operation_omission": _candidates_omission,
                "reorder": _candidates_reorder}


# =====================================================================
# Geometry difference classification
# =====================================================================

@dataclass(frozen=True)
class GeoDiff:
    changed: bool
    kind: str        # topology | numeric | exec_error | none
    detail: str


_TOPO_KEYS = ("is_valid", "solid_count", "shell_count", "face_count",
              "edge_count", "vertex_count", "genus", "loop_count")
_NUM_KEYS = ("volume", "area", "bbox_x", "bbox_y", "bbox_z")


def props_to_dict(props: GeoProps | None) -> dict | None:
    if props is None:
        return None
    return {"volume": props.volume, "area": props.area,
            "bbox_x": props.bbox.x, "bbox_y": props.bbox.y, "bbox_z": props.bbox.z,
            "vertex_count": props.vertex_count, "edge_count": props.edge_count,
            "face_count": props.face_count, "shell_count": props.shell_count,
            "solid_count": props.solid_count, "loop_count": props.loop_count,
            "genus": props.genus, "is_valid": props.is_valid}


def geo_diff(gold: dict | None, inj: dict | None, rel_tol: float = REL_TOL) -> GeoDiff:
    if gold is None:
        raise ValueError("gold props is None — gold must pass the executability gate first")
    if inj is None:
        return GeoDiff(True, "exec_error", "injected produced no solid")
    for key in _TOPO_KEYS:
        if gold[key] != inj[key]:
            return GeoDiff(True, "topology", f"{key} {gold[key]}->{inj[key]}")
    for key in _NUM_KEYS:
        a, b = gold[key], inj[key]
        if abs(a - b) > rel_tol * max(abs(a), 1e-9):
            return GeoDiff(True, "numeric", f"{key} {a:.6g}->{b:.6g}")
    return GeoDiff(False, "none", "geometry unchanged")


# =====================================================================
# Isolated geometry worker
# =====================================================================

@contextlib.contextmanager
def _silence_native_stdout():
    """Redirect native stdout so it cannot corrupt the worker result frame."""
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def _final_ok_props(code: str, step_timeout: float) -> GeoProps | None:
    snaps = execute_steps(split_steps(code), timeout=step_timeout)
    for s in reversed(snaps):
        if s.ok and s.props is not None:
            return s.props
    return None


def _worker_main() -> None:
    import traceback
    raw = sys.stdin.read()
    try:
        with _silence_native_stdout():
            req = json.loads(raw)
            props = _final_ok_props(req["code"], float(req.get("step_timeout",
                                                                DEFAULT_STEP_TIMEOUT)))
            out = {"props": props_to_dict(props)}
    except Exception:  # noqa: BLE001
        out = {"worker_error": traceback.format_exc()}
    payload = RESULT_SENTINEL + json.dumps(_json_safe(out), ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def run_props_worker(code: str, *, step_timeout: float = DEFAULT_STEP_TIMEOUT,
                     wall_timeout: float = DEFAULT_WALL_TIMEOUT) -> tuple[dict | None, str]:
    """Return final geometry properties and worker status."""
    # A subprocess isolates native OCCT crashes and provides a hard wall clock.
    req = json.dumps({"code": code, "step_timeout": step_timeout}, ensure_ascii=False)
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
    return data["props"], "ok"


def run_gold_props_cached(gold_code: str, *, step_timeout: float = DEFAULT_STEP_TIMEOUT,
                          wall_timeout: float = DEFAULT_WALL_TIMEOUT) -> tuple[dict | None, str]:
    """Evaluate a reference program in an isolated worker process.

    The public release omits the private workspace cache. The historical
    function name is retained so the mutation algorithm is unchanged.
    """
    return run_props_worker(gold_code, step_timeout=step_timeout,
                            wall_timeout=wall_timeout)


# =====================================================================
# Mutation selection and geometry filtering
# =====================================================================

@dataclass
class InjectionResult:
    injected_code: str
    injected_step_index: int
    error_class: str
    validity_kind: str           # topology | numeric | exec_error
    detail: str
    mutation: dict


def inject(gold_code: str, error_class: str, seed: int = 0, *,
           step_timeout: float = DEFAULT_STEP_TIMEOUT,
           wall_timeout: float = DEFAULT_WALL_TIMEOUT,
           gold_props: dict | None = None,
           gold_status: str | None = None) -> tuple[InjectionResult | None, str]:
    """Inject one error class and return a result plus status reason."""
    try:
        tree = ast.parse(gold_code)
    except SyntaxError:
        return None, "gold_unparseable"
    if _has_control_flow(tree):
        return None, "control_flow"
    if not _enumerate_terminals(tree):
        return None, "no_eligible_site"
    assert_step_consistency(gold_code)

    cands = CANDIDATE_FN[error_class](tree)
    if not cands:
        return None, "no_eligible_site"
    rng = random.Random(seed)
    start = rng.randrange(len(cands))
    order = [cands[(start + i) % len(cands)] for i in range(len(cands))]

    # Syntax mutations need no geometry worker.
    if error_class == "syntax_error":
        for (cand_code, owner, mut) in order:
            if not _parses(cand_code):
                return InjectionResult(cand_code, owner, error_class, "exec_error",
                                       "unparseable (syntax injection)", mut), "ok"
        return None, "harmless_exhausted"

    if gold_props is None and gold_status is None:
        gold_props, gold_status = run_gold_props_cached(gold_code, step_timeout=step_timeout,
                                                        wall_timeout=wall_timeout)
    if gold_status == "worker_error":
        return None, "infra_error"
    if gold_props is None:
        return None, "gold_unusable"

    for (cand_code, owner, mut) in order:
        cprops, cstatus = run_props_worker(cand_code, step_timeout=step_timeout,
                                           wall_timeout=wall_timeout)
        if cstatus in ("timeout", "crash") or cprops is None:
            diff = GeoDiff(True, "exec_error", cstatus if cstatus != "ok" else "no_solid")
        else:
            diff = geo_diff(gold_props, cprops)
        if diff.changed:
            return InjectionResult(cand_code, owner, error_class, diff.kind,
                                   diff.detail, mut), "ok"
    return None, "harmless_exhausted"


# =====================================================================
# Batch generation
# =====================================================================

def main(inputs: str, out_dir: str, classes: list, n: int, seed: int,
         step_timeout: float, wall_timeout: float) -> None:
    rows = [json.loads(line) for line in
            Path(inputs).read_text(encoding="utf-8").splitlines() if line.strip()]
    if n > 0:
        rows = rows[:n]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    samples: list = []
    meta = {c: {"attempted": 0, "accepted": 0, "harmless_exhausted": 0,
                "no_eligible_site": 0, "gold_unusable": 0, "control_flow": 0,
                "infra_error": 0, "topology": 0, "numeric": 0, "exec_error": 0}
            for c in classes}

    for i, row in enumerate(rows, 1):
        gold = row.get("gold_code")
        rid = str(row.get("id", i))
        if not gold:
            continue
        gprops, gstatus = run_gold_props_cached(gold, step_timeout=step_timeout,
                                                wall_timeout=wall_timeout)
        for c in classes:
            meta[c]["attempted"] += 1
            res, reason = inject(gold, c, seed, step_timeout=step_timeout,
                                 wall_timeout=wall_timeout, gold_props=gprops,
                                 gold_status=gstatus)
            if res is None:
                meta[c][reason] = meta[c].get(reason, 0) + 1
                continue
            meta[c]["accepted"] += 1
            meta[c][res.validity_kind] = meta[c].get(res.validity_kind, 0) + 1
            samples.append({"id": f"{rid}__{c}", "source_id": rid, "error_class": c,
                            "injected_code": res.injected_code, "gold_code": gold,
                            "injected_step_index": res.injected_step_index,
                            "truth_steps": res.mutation.get("truth_steps",
                                                            [res.injected_step_index]),
                            "truth_frame": res.mutation.get("truth_frame", "injected"),
                            "validity_kind": res.validity_kind, "geo_detail": res.detail,
                            "mutation": res.mutation})
        print(f"[{i}/{len(rows)}] {rid:12s} "
              + " ".join(f"{c}={('ok' if any(s['source_id']==rid and s['error_class']==c for s in samples) else '--')}"
                         for c in classes))

    (out / "injected_samples.jsonl").write_text(
        "\n".join(json.dumps(_json_safe(s), ensure_ascii=False) for s in samples) + "\n",
        encoding="utf-8")
    for c in classes:
        acc, har = meta[c]["accepted"], meta[c]["harmless_exhausted"]
        meta[c]["harmless_reject_rate"] = (har / (acc + har)) if (acc + har) else None
    meta["_total"] = {"rows": len(rows), "samples": len(samples)}
    (out / "injection_meta.json").write_text(
        json.dumps(_json_safe(meta), ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== INJECTION META =====")
    print(json.dumps(_json_safe(meta), ensure_ascii=False, indent=2))


# =====================================================================
# Smoke tests
# =====================================================================

MULTI_STMT = ("import cadquery as cq\n"
              "result = cq.Workplane('XY').box(100, 60, 40)\n"
              "result = result.faces('>Z').workplane().circle(10).cutThruAll()\n"
              "result = result.edges('|Z').fillet(5)\n")
MANY_TO_ONE = ("import cadquery as cq\n"
               "result = cq.Workplane('XY').box(40, 40, 40)\n"
               "result = result.faces('>Z').circle(8).cutThruAll().faces('<Z').hole(4)\n")
TWO_BODY = ("import cadquery as cq\n"
            "part_1 = cq.Workplane('XY').box(10, 10, 10)\n"
            "part_2 = cq.Workplane('XY').box(4, 4, 20)\n"
            "part_1 = part_1.cut(part_2)\n")
TUPLE_ARG = ("import cadquery as cq\n"
             "r = cq.Workplane('XY').box(1, 1, 1)\n"
             "r = r.translate((0, 0.1759, 0))\n")


def _ast_tests() -> None:
    print("===== MUTATION AST SMOKE TESTS =====")
    assert [s.op_name for s in _enumerate_terminals(ast.parse(MULTI_STMT))] == \
        ["box", "cutThruAll", "fillet"]
    assert_step_consistency(MULTI_STMT)
    assert [(s.op_name, s.step_index) for s in _enumerate_terminals(ast.parse(MANY_TO_ONE))] == \
        [("box", 1), ("cutThruAll", 2), ("hole", 3)]
    assert_step_consistency(MANY_TO_ONE)
    print("  [PASS] mutation and checker step enumeration agree")

    ps = _candidates_param_shift(ast.parse(TUPLE_ARG))
    assert any(abs(d["from"] - 0.1759) < 1e-9 for _, _, d in ps)
    print("  [PASS] nested tuple arguments are mutation candidates")

    demo_buggy = ("import cadquery as cq\n"
                  "result = cq.Workplane('XY').box(100, 60, 40)\n"
                  "result = result.faces('>Z').workplane().circle(10).cutBlind(5)\n")
    ps2 = _candidates_param_shift(ast.parse(demo_buggy))
    neg = [(c, o) for c, o, d in ps2 if d["mode"] == "negate" and d["from"] == 5.0]
    assert neg and neg[0][1] == 2 and "cutBlind(-5)" in neg[0][0]
    print("  [PASS] numeric negation labels the correct step")

    bf = _candidates_boolean_flip(ast.parse(TWO_BODY))
    cu = [(c, o, d) for c, o, d in bf if d.get("sub") == "method" and d["from"] == "cut"]
    assert cu and cu[0][2]["to"] == "union" and "union(part_2)" in cu[0][0]
    print("  [PASS] boolean method substitutions")

    sels = [d for _, _, d in _candidates_boolean_flip(ast.parse(MULTI_STMT)) if d.get("sub") == "selector"]
    assert any(d["from"] == ">Z" for d in sels) and not any(d["from"] == "|Z" for d in sels)
    print("  [PASS] directional selector substitutions")

    am = _candidates_api_misuse(ast.parse(MULTI_STMT))
    assert any(o == 2 and d.get("to") == "cutBlind" for _, o, d in am)
    print("  [PASS] compound API substitutions")

    sx = _candidates_syntax(ast.parse(MULTI_STMT))
    assert sx and all(not _parses(c) for c, _, _ in sx)
    assert any(d["line"] == 2 and d["truth_steps"] == [2] for _, _, d in sx)
    print("  [PASS] syntax mutations are invalid and correctly labeled")

    om = _candidates_omission(ast.parse(MANY_TO_ONE))
    dc = [(c, o, d) for c, o, d in om
          if d["sub"] == "drop_op" and d["removed"] == "cutThruAll"]
    assert dc
    c0, o0, d0 = dc[0]
    assert _parses(c0) and "cutThruAll" not in c0 and ".hole(4)" in c0
    assert d0["gold_step"] == 2 and o0 == 2 and d0["truth_steps"] == [2]
    print("  [PASS] operation omission preserves surrounding calls")

    om2 = _candidates_omission(ast.parse(MULTI_STMT))
    fd = [(c, o, d) for c, o, d in om2 if d["sub"] == "drop_stmt" and "fillet" not in c]
    assert fd and fd[0][2]["gold_step"] == 3 and fd[0][1] == 2
    print("  [PASS] statement omission labels the first divergence")

    ro = _candidates_reorder(ast.parse(MULTI_STMT))
    assert any(d["truth_steps"] == [2, 3] for _, _, d in ro)
    assert not any(d["swapped"] == [1, 2] for _, _, d in ro)
    print("  [PASS] reordering respects first definitions")

    assert [c for c, _, _ in _candidates_param_shift(ast.parse(MULTI_STMT))] == \
        [c for c, _, _ in _candidates_param_shift(ast.parse(MULTI_STMT))]
    assert [c for c, _, _ in _candidates_omission(ast.parse(MULTI_STMT))] == \
        [c for c, _, _ in _candidates_omission(ast.parse(MULTI_STMT))]
    print("  [PASS] candidate generation is deterministic")

    g = {"is_valid": True, "solid_count": 1, "shell_count": 1, "face_count": 6,
         "edge_count": 12, "vertex_count": 8, "genus": 0, "loop_count": 6,
         "volume": 1000.0, "area": 600.0, "bbox_x": 10, "bbox_y": 10, "bbox_z": 10}
    topo = dict(g, face_count=11, genus=1)
    num = dict(g, volume=100.0)
    assert geo_diff(g, topo).kind == "topology"
    assert geo_diff(g, num).kind == "numeric"
    assert geo_diff(g, None).kind == "exec_error"
    assert geo_diff(g, dict(g)).changed is False
    print("  [PASS] geometry difference classes")


def _occt_tests() -> None:
    from _samples import FIXED_CODE as DEMO_GOLD
    print("\n===== MUTATION GEOMETRY SMOKE TESTS =====")

    gp, gs = run_props_worker(DEMO_GOLD)
    assert gs == "ok" and gp is not None, f"worker preflight failed: status={gs}"
    print(f"  [PASS] reference program executed (face_count={gp['face_count']})")

    res, reason = inject(DEMO_GOLD, "api_misuse", seed=0)
    assert res is not None and res.injected_step_index in (2, 3), f"api_misuse: {reason}"
    print(f"  [PASS] API mutation: step={res.injected_step_index} kind={res.validity_kind}")

    res3, r3 = inject(DEMO_GOLD, "param_shift", seed=0)
    assert res3 is not None, f"param_shift: {r3}"
    print(f"  [PASS] parameter mutation: step={res3.injected_step_index} kind={res3.validity_kind}")

    res6, r6 = inject(DEMO_GOLD, "boolean_flip", seed=0)
    assert res6 is None and r6 == "harmless_exhausted", f"unexpected result: {r6}"
    print("  [PASS] geometrically harmless mutations are rejected")

    rs, rr = inject(DEMO_GOLD, "syntax_error", seed=0)
    assert rs is not None and rs.validity_kind == "exec_error" \
        and not _parses(rs.injected_code), f"syntax_error: {rr}"
    print(f"  [PASS] syntax mutation: truth={rs.mutation['truth_steps']}")

    ro4, r4 = inject(DEMO_GOLD, "operation_omission", seed=0)
    assert ro4 is not None, f"operation_omission: {r4}"
    print(f"  [PASS] operation omission: step={ro4.injected_step_index} kind={ro4.validity_kind}")

    # Reordering cut and union changes this geometry.
    order_gold = ("import cadquery as cq\n"
                  "a = cq.Workplane('XY').box(40, 40, 10)\n"
                  "b = cq.Workplane('XY').cylinder(30, 6)\n"
                  "a = a.cut(b)\n"
                  "a = a.union(cq.Workplane('XY').box(10, 10, 30))\n")
    ro5, r5 = inject(order_gold, "reorder", seed=0)
    assert ro5 is not None and ro5.mutation["truth_steps"] == [3, 4], f"reorder: {r5}"
    print(f"  [PASS] reorder mutation: truth={ro5.mutation['truth_steps']} "
          f"kind={ro5.validity_kind}")

    print("\nMutation smoke tests passed.")


def smoke() -> None:
    _ast_tests()
    _occt_tests()


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

    ap = argparse.ArgumentParser(description="Generate controlled CadQuery mutations")
    ap.add_argument("--smoke", action="store_true", help="run AST and geometry smoke tests")
    ap.add_argument("--inputs")
    ap.add_argument("--out-dir")
    ap.add_argument("--classes", nargs="+", default=list(ERROR_CLASSES), choices=list(ERROR_CLASSES))
    ap.add_argument("--n", type=int, default=0, help="maximum rows; 0 means all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step-timeout", type=float, default=DEFAULT_STEP_TIMEOUT)
    ap.add_argument("--timeout", type=float, default=DEFAULT_WALL_TIMEOUT, help="worker timeout in seconds")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    else:
        if not args.inputs:
            ap.error("batch mode requires --inputs (or use --smoke)")
        if not args.out_dir:
            ap.error("batch mode requires --out-dir (or use --smoke)")
        main(inputs=args.inputs, out_dir=args.out_dir, classes=args.classes, n=args.n,
             seed=args.seed, step_timeout=args.step_timeout, wall_timeout=args.timeout)
