"""Synthesize self-consistent assertions from a reference CAD program.

The reference program is probed in an isolated subprocess. Assertions are
derived from measured topology, numeric properties, deltas, and detected
symmetries, then checked against the same reference. Assertions that fail this
self-consistency check are removed before the result is returned.

Example: python synthesize_assertions.py --smoke --no-symmetry
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THIS_FILE = Path(__file__).resolve()
from geoassert import (
    check, check_symmetry, execute_steps, split_steps,
)
from geoassert.constants import NUMERIC_TOL_PCT as _NUMERIC_TOL_PCT

RESULT_SENTINEL = b"@@GEOASSERT_PROBE_RESULT@@"
DEFAULT_STEP_TIMEOUT = 30.0
DEFAULT_WALL_TIMEOUT = 180.0
NUMERIC_TOL_PCT = _NUMERIC_TOL_PCT
SYM_PLANES = ("XY", "XZ", "YZ")


# =====================================================================
# Serialization helpers
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


def _props_to_dict(props) -> dict | None:
    if props is None:
        return None
    return {"volume": props.volume, "area": props.area,
            "bbox_x": props.bbox.x, "bbox_y": props.bbox.y, "bbox_z": props.bbox.z,
            "vertex_count": props.vertex_count, "edge_count": props.edge_count,
            "face_count": props.face_count, "shell_count": props.shell_count,
            "solid_count": props.solid_count, "loop_count": props.loop_count,
            "genus": props.genus, "is_valid": props.is_valid}


def _scope_text(scope: int | None) -> str:
    return "final" if scope is None else f"step={scope}"


# =====================================================================
# Isolated geometry worker
# =====================================================================

def _do_probe(code: str, step_timeout: float, symmetry: bool, planes) -> dict:
    snaps = execute_steps(split_steps(code), timeout=step_timeout)
    steps = [{"index": s.step.index, "label": s.step.label, "ok": s.ok,
              "props": _props_to_dict(s.props)} for s in snaps]
    sym, sym_status = None, "skipped"
    if symmetry:
        final = next((s for s in reversed(snaps) if s.ok and s.shape is not None), None)
        if final is None:
            sym_status = "no_shape"
        else:
            sym, sym_status = {}, "ok"
            for plane in planes:
                try:
                    ok, dev = check_symmetry(final.shape, plane)
                    sym[plane] = [bool(ok), float(dev)]
                except Exception:  # noqa: BLE001 - retain results from other planes
                    sym[plane] = None
                    sym_status = "error"
    return {"steps": steps, "symmetry": sym, "sym_status": sym_status}


def _do_check(code: str, cot: str, timeout: float) -> dict:
    rep = check(code, cot, timeout=timeout)
    return {
        "ok": bool(rep.ok),
        "execution_error": (rep.execution_error.step.index
                            if rep.execution_error is not None else None),
        "failures": [{"step": (r.step.index if r.step else None),
                      "scope": r.assertion.scope_text, "raw": r.assertion.raw}
                     for r in rep.failures],
    }


def _worker_main() -> None:
    import traceback
    raw = sys.stdin.read()
    try:
        with _silence_native_stdout():
            req = json.loads(raw)
            task = req.get("task")
            if task == "probe":
                out = _do_probe(req["code"], float(req.get("step_timeout", DEFAULT_STEP_TIMEOUT)),
                                bool(req.get("symmetry", True)),
                                tuple(req.get("planes", SYM_PLANES)))
            elif task == "check":
                out = _do_check(req["code"], req["cot"],
                                float(req.get("timeout", DEFAULT_STEP_TIMEOUT)))
            else:
                out = {"worker_error": f"unknown task {task!r}"}
    except Exception:  # noqa: BLE001
        out = {"worker_error": traceback.format_exc()}
    payload = RESULT_SENTINEL + json.dumps(_json_safe(out), ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _run_worker(req: dict, wall_timeout: float) -> tuple[dict | None, str]:
    # A subprocess isolates native OCCT crashes and enforces a wall clock.
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        proc = subprocess.run([sys.executable, str(THIS_FILE), "--worker"],
                              input=json.dumps(req, ensure_ascii=False).encode("utf-8"),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=wall_timeout, **kwargs)
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


# =====================================================================
# Probe result structures
# =====================================================================

@dataclass
class StepProbe:
    index: int
    label: str
    ok: bool
    props: dict | None


@dataclass
class ProbeResult:
    steps: list
    symmetry: dict | None
    sym_status: str

    def solid_steps(self) -> list:
        return [s for s in self.steps if s.ok and s.props is not None]

    @property
    def first_solid_step(self) -> StepProbe | None:
        ss = self.solid_steps()
        return ss[0] if ss else None


def _probe_from_data(data: dict) -> ProbeResult:
    steps = [StepProbe(s["index"], s["label"], s["ok"], s["props"]) for s in data["steps"]]
    return ProbeResult(steps, data.get("symmetry"), data.get("sym_status", "skipped"))


def run_probe_worker(code: str, *, step_timeout: float = DEFAULT_STEP_TIMEOUT,
                     wall_timeout: float = DEFAULT_WALL_TIMEOUT,
                     symmetry: bool = True,
                     cache_gold: bool = False) -> tuple[ProbeResult | None, str]:
    """Probe a program in an isolated worker process.

    ``cache_gold`` is accepted for compatibility; the public release does not
    persist private-workspace caches.
    """
    data, status = _run_worker(
        {"task": "probe", "code": code, "step_timeout": step_timeout,
         "symmetry": symmetry, "planes": list(SYM_PLANES)}, wall_timeout)
    if status != "ok":
        return None, status
    return _probe_from_data(data), "ok"


def run_check_worker(code: str, cot: str, *, timeout: float = DEFAULT_STEP_TIMEOUT,
                     wall_timeout: float = DEFAULT_WALL_TIMEOUT) -> tuple[dict | None, str]:
    return _run_worker({"task": "check", "code": code, "cot": cot, "timeout": timeout},
                       wall_timeout)


# =====================================================================
# Assertion synthesis
# =====================================================================

def _emit_step(idx: int, cur: dict, prev: dict | None,
               tol_pct: float = NUMERIC_TOL_PCT) -> list:
    """Return assertions for one step that produced a solid."""
    out: list = []
    add = lambda body: out.append((idx, body))  # noqa: E731
    add(f"bbox=({cur['bbox_x']:.6g},{cur['bbox_y']:.6g},{cur['bbox_z']:.6g}) tol={tol_pct:g}%")
    add(f"volume={cur['volume']:.6g} tol={tol_pct:g}%")
    add(f"face_count={cur['face_count']}")
    add(f"solid_count={cur['solid_count']}")
    if cur["genus"] is not None:
        add(f"through_holes={cur['genus']}")
    if cur["is_valid"]:
        add("is_valid=true")
    if prev is not None:
        dv = cur["volume"] - prev["volume"]
        eps = 1e-6 * max(prev["volume"], 1.0)
        if dv > eps:
            add("delta_volume=increase")
        elif dv < -eps:
            add("delta_volume=decrease")
        df = cur["face_count"] - prev["face_count"]
        if df > 0:
            add("delta_faces=increase")
        elif df < 0:
            add("delta_faces=decrease")
    return out


def _emit_final(final: dict, symmetry: dict | None) -> list:
    out: list = []
    add = lambda body: out.append((None, body))  # noqa: E731
    if final["genus"] is not None:
        add(f"through_holes={final['genus']}")
    if final["solid_count"] >= 1 and final["is_valid"]:
        add("is_watertight=true")
    bbox_vol = final["bbox_x"] * final["bbox_y"] * final["bbox_z"]
    if final["volume"] < 0.95 * bbox_vol:
        add("volume < 0.95 * bbox_vol")
    if symmetry:
        planes = sorted(p for p, v in symmetry.items() if v and v[0])
        if planes:
            add(f"symmetry={','.join(planes)}")
    return out


def synthesize_from_probe(probe: ProbeResult,
                          tol_pct: float = NUMERIC_TOL_PCT) -> list:
    """Return ordered ``(scope, body)`` assertion pairs."""
    solids = probe.solid_steps()
    by_index = {s.index: s for s in solids}
    asserts: list = []
    for sp in solids:
        prev = by_index.get(sp.index - 1)
        asserts += _emit_step(sp.index, sp.props, prev.props if prev else None,
                              tol_pct=tol_pct)
    if solids:
        asserts += _emit_final(solids[-1].props, probe.symmetry)
    return asserts


def render_cot(asserts: list) -> str:
    return "".join(f"@assert {_scope_text(sc)} {body}\n" for sc, body in asserts)


# =====================================================================
# Self-consistency filtering
# =====================================================================

@dataclass
class SynthResult:
    gold_code: str
    ok: bool
    cot: str = ""
    gold_assertions: list = field(default_factory=list)
    drop_reason: str | None = None
    n_assertions: int = 0
    n_dropped: int = 0
    has_symmetry: bool = False
    sym_status: str = "skipped"


def synthesize(gold_code: str, *, symmetry: bool = True,
               step_timeout: float = DEFAULT_STEP_TIMEOUT,
               wall_timeout: float = DEFAULT_WALL_TIMEOUT,
               max_repair_rounds: int = 3,
               numeric_tol_pct: float = NUMERIC_TOL_PCT) -> SynthResult:
    # Retry without symmetry if the more expensive probe fails.
    probe, status = run_probe_worker(gold_code, step_timeout=step_timeout,
                                     wall_timeout=wall_timeout, symmetry=symmetry,
                                     cache_gold=True)
    if status != "ok" and symmetry:
        probe, status = run_probe_worker(gold_code, step_timeout=step_timeout,
                                         wall_timeout=wall_timeout, symmetry=False,
                                         cache_gold=True)
    if status != "ok" or probe.first_solid_step is None:
        return SynthResult(gold_code, ok=False, drop_reason="gold_unexecutable")

    asserts = synthesize_from_probe(probe, tol_pct=numeric_tol_pct)
    n_drop = 0
    ok = False
    for rnd in range(max_repair_rounds + 1):
        cot = render_cot(asserts)
        rep, cstatus = run_check_worker(gold_code, cot, timeout=step_timeout,
                                        wall_timeout=wall_timeout)
        if cstatus != "ok":
            return SynthResult(gold_code, ok=False, drop_reason="gold_check_infra",
                               sym_status=probe.sym_status)
        if rep["ok"]:
            ok = True
            break
        if rnd == max_repair_rounds:
            break
        failed = {(f["scope"], f["raw"]) for f in rep["failures"]}
        kept = [(sc, body) for (sc, body) in asserts if (_scope_text(sc), body) not in failed]
        if len(kept) == len(asserts):
            break
        n_drop += len(asserts) - len(kept)
        asserts = kept

    cot = render_cot(asserts)
    return SynthResult(
        gold_code, ok=ok, cot=cot,
        gold_assertions=[body for _, body in asserts],
        drop_reason=None if ok else "gold_self_inconsistent",
        n_assertions=len(asserts), n_dropped=n_drop,
        has_symmetry=any(body.startswith("symmetry=") for _, body in asserts),
        sym_status=probe.sym_status)


# =====================================================================
# Smoke tests
# =====================================================================

def smoke(*, symmetry: bool = True) -> None:
    print("===== ASSERTION SYNTHESIS SMOKE TESTS =====")

    # Verify deterministic formatting without invoking OCCT.
    fake = ProbeResult(steps=[StepProbe(1, "box", True, {
        "volume": 1000.0, "area": 600.0, "bbox_x": 10.0, "bbox_y": 10.0, "bbox_z": 10.0,
        "vertex_count": 8, "edge_count": 12, "face_count": 6, "shell_count": 1,
        "solid_count": 1, "loop_count": 6, "genus": 0, "is_valid": True})],
        symmetry=None, sym_status="skipped")
    cot0 = render_cot(synthesize_from_probe(fake))
    assert "@assert step=1 face_count=6" in cot0
    assert "@assert step=1 through_holes=0" in cot0
    assert "@assert step=1 volume=1000 tol=2%" in cot0
    assert "@assert final is_watertight=true" in cot0
    print("  [PASS] deterministic assertion formatting")

    # Verify synthesized assertions against an executed reference program.
    from _samples import FIXED_CODE
    sr = synthesize(FIXED_CODE, symmetry=symmetry)
    assert sr.ok, f"synth not ok: {sr.drop_reason}"
    assert "face_count=" in sr.cot and "through_holes=1" in sr.cot
    assert "delta_volume=decrease" in sr.cot, "cutting the hole should reduce volume"
    rep, st = run_check_worker(FIXED_CODE, sr.cot)
    assert st == "ok" and rep["ok"], "reference assertions are not self-consistent"
    print(f"  [PASS] reference assertions: {sr.n_assertions} retained, "
          f"{sr.n_dropped} dropped, symmetry={sr.has_symmetry} ({sr.sym_status})")
    print("--- SYNTHESIZED ASSERTIONS ---")
    print(sr.cot)

    print("Assertion synthesis smoke tests passed.")


if __name__ == "__main__":
    if "--worker" in sys.argv[1:]:
        _worker_main()
        sys.exit(0)

    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Synthesize assertions from reference geometry")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-symmetry", action="store_true", help="skip symmetry checks")
    args = ap.parse_args()
    if args.smoke:
        smoke(symmetry=not args.no_symmetry)
    else:
        ap.error("use --smoke here; use gate_samples.py for batch processing")
