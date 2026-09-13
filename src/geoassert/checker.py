"""Execution-grounded assertions for generated CadQuery programs.

Verifies machine-checkable geometric assertions embedded in a model's
chain-of-thought against the *actual* geometry produced by executing the
generated CadQuery code, step by step.

Pipeline
--------
1.  ``split_steps``     : AST-split CadQuery source into cumulative,
                          executable prefixes (one per terminal 3D operation).
2.  ``execute_steps``   : sandbox-execute every prefix, snapshot geometric
                          properties (volume, bbox, topology, genus, ...).
3.  ``parse_assertions``: parse ``@assert`` lines from a CoT string.
4.  ``check``           : evaluate every assertion against the snapshots and
                          produce a localized ``Report``.
5.  ``build_repair_prompt``: turn failures into a targeted repair prompt.

Quick start
-----------
>>> from geoassert import check
>>> report = check(code, cot)          # cot contains @assert lines
>>> print(report.render())
>>> if not report.ok:
...     print(build_repair_prompt(code, report))

Assertion DSL
-------------
``@assert [step=K | final] <body>``  (scope defaults to ``final``)

key=value bodies::

    volume=240000 tol=10%        area=52000 tol=5%
    bbox=(100,60,40) tol=5%      bbox_sorted=(40,60,100) tol=5%
    face_count=7   edge_count=15  vertex_count=10   solid_count=1
    through_holes=1              is_valid=true      is_watertight=true
    delta_volume=decrease        delta_volume=-12566 tol=10%
    delta_faces=+1               delta_faces=increase
    symmetry=XZ,YZ

relational bodies (safe-evaluated against the measured properties)::

    @assert final bbox.x > bbox.y
    @assert final volume < 0.8 * bbox_vol

Genus / through-hole counting uses the full Euler–Poincaré relation for
B-rep solids,  ``V − E + F − (L − F) = 2(S − G)``  with degenerate edges
(e.g. sphere poles) excluded from ``E`` — empirically validated on
genus-0/1/2 shapes including spheres, cones and tori.

Dependencies are declared in ``pyproject.toml``. The tested Python range is
3.11--3.12.

This is a research sandbox, not a security boundary: generated code is
executed in-process with a wall-clock timeout.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import math
import re
import signal
import sys
import threading
import traceback
from dataclasses import dataclass
from typing import Any

import cadquery as cq
import numpy as np
from OCP.BRep import BRep_Tool
from OCP.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_IndexedMapOfShape

__all__ = [
    "BBox",
    "GeoProps",
    "Step",
    "Snapshot",
    "Assertion",
    "CheckResult",
    "Report",
    "split_steps",
    "execute_steps",
    "parse_assertions",
    "check",
    "build_repair_prompt",
]

# =====================================================================
# 1. Geometric properties
# =====================================================================


@dataclass(frozen=True)
class BBox:
    x: float
    y: float
    z: float

    @property
    def volume(self) -> float:
        return self.x * self.y * self.z

    @property
    def diagonal(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def sorted(self) -> tuple[float, float, float]:
        return tuple(sorted((self.x, self.y, self.z)))  # type: ignore[return-value]


@dataclass(frozen=True)
class GeoProps:
    """Snapshot of the measurable geometry of one execution prefix."""

    volume: float
    area: float
    bbox: BBox
    center: tuple[float, float, float]
    vertex_count: int
    edge_count: int          # non-degenerate edges
    face_count: int
    shell_count: int
    solid_count: int
    loop_count: int          # total per-face wire count
    genus: int | None     # None if Euler relation is non-integral
    is_valid: bool

    # -- derived ------------------------------------------------------
    @property
    def through_holes(self) -> int | None:
        """Genus of the solid == number of through holes (handles)."""
        return self.genus

    @property
    def is_watertight(self) -> bool:
        return self.solid_count >= 1 and self.is_valid

    def namespace(self) -> dict[str, Any]:
        """Names exposed to relational assertion expressions."""
        return {
            "volume": self.volume,
            "area": self.area,
            "bbox": self.bbox,
            "bbox_vol": self.bbox.volume,
            "vertex_count": self.vertex_count,
            "edge_count": self.edge_count,
            "face_count": self.face_count,
            "solid_count": self.solid_count,
            "through_holes": self.genus,
            "genus": self.genus,
        }

    # -- construction --------------------------------------------------
    @staticmethod
    def from_shape(shape: cq.Shape) -> GeoProps:
        def umap(kind) -> TopTools_IndexedMapOfShape:
            m = TopTools_IndexedMapOfShape()
            TopExp.MapShapes_s(shape.wrapped, kind, m)
            return m

        v = umap(TopAbs_VERTEX).Extent()
        f = umap(TopAbs_FACE).Extent()
        s = umap(TopAbs_SHELL).Extent()
        n_solids = umap(TopAbs_SOLID).Extent()

        edge_map = umap(TopAbs_EDGE)
        e = sum(
            1
            for i in range(1, edge_map.Extent() + 1)
            if not BRep_Tool.Degenerated_s(TopoDS.Edge_s(edge_map.FindKey(i)))
        )

        loops = 0
        for face in shape.Faces():
            exp = TopExp_Explorer(face.wrapped, TopAbs_WIRE)
            while exp.More():
                loops += 1
                exp.Next()

        # Euler–Poincaré for B-rep:  V − E + F − (L − F) = 2(S − G)
        g_float = s - (v - e + f - (loops - f)) / 2
        genus = int(round(g_float)) if abs(g_float - round(g_float)) < 1e-9 else None

        bb = shape.BoundingBox()
        c = shape.Center()
        return GeoProps(
            volume=shape.Volume(),
            area=shape.Area(),
            bbox=BBox(bb.xlen, bb.ylen, bb.zlen),
            center=(c.x, c.y, c.z),
            vertex_count=v,
            edge_count=e,
            face_count=f,
            shell_count=s,
            solid_count=n_solids,
            loop_count=loops,
            genus=genus,
            is_valid=shape.isValid(),
        )


def _to_shape(obj: Any) -> cq.Shape | None:
    """Best-effort extraction of a solid Shape from an executed object."""
    if isinstance(obj, cq.Shape):
        return obj if obj.Solids() else None
    if isinstance(obj, cq.Workplane):
        with contextlib.suppress(ValueError):
            return obj.findSolid()
    return None


# --- symmetry (mesh-sampled reflection test) --------------------------

_PLANE_NORMAL_AXIS = {"XY": 2, "XZ": 1, "YZ": 0}  # axis flipped by reflection

# On Windows, trimesh proximity queries can crash for OCCT-derived meshes, so
# symmetry checking uses a sampled cKDTree approximation on that platform.
_TRIMESH_PROXIMITY_OK = sys.platform != "win32"

# A fixed seed makes the symmetry measurement deterministic.
SYMMETRY_SAMPLE_SEED = 20260708
# Relative threshold plus an absolute floor for floating-point noise.
DELTA_VOL_EPS_REL = 1e-6
DELTA_VOL_EPS_ABS = 1e-12


def _sample_surface_seeded(mesh, n: int, seed: int):
    """Sample a mesh surface deterministically across trimesh versions."""
    import trimesh

    try:
        return trimesh.sample.sample_surface(mesh, n, seed=seed)
    except TypeError:
        state = np.random.get_state()
        np.random.seed(seed % (2**32))
        try:
            return trimesh.sample.sample_surface(mesh, n)
        finally:
            np.random.set_state(state)


def check_symmetry(
    shape: cq.Shape, plane: str, n_samples: int = 1500, rel_tol: float = 0.01
) -> tuple[bool, float]:
    """Test mirror symmetry across a bbox-center plane ("XY" | "XZ" | "YZ").

    Samples surface points, reflects them across the plane, and measures
    the 98th-percentile *exact point-to-mesh* distance relative to the
    bbox diagonal.  Using exact surface distance (rather than nearest
    sampled point) makes the test independent of sampling density: for a
    truly symmetric shape every reflected point lies on the surface.

    Returns ``(symmetric, relative_deviation)``.
    """
    import trimesh

    axis = _PLANE_NORMAL_AXIS[plane.upper()]
    bb = shape.BoundingBox()
    diag = math.sqrt(bb.xlen**2 + bb.ylen**2 + bb.zlen**2)

    verts, tris = shape.tessellate(max(diag / 1000.0, 1e-3))
    mesh = trimesh.Trimesh(
        vertices=[(p.x, p.y, p.z) for p in verts], faces=tris, process=False
    )
    pts, _ = _sample_surface_seeded(mesh, n_samples, SYMMETRY_SAMPLE_SEED)
    center = mesh.bounds.mean(axis=0)

    reflected = pts.copy()
    reflected[:, axis] = 2 * center[axis] - reflected[:, axis]

    if _TRIMESH_PROXIMITY_OK:
        # Exact distance to the triangle mesh.
        _, dist, _ = trimesh.proximity.closest_point(mesh, reflected)
    else:
        # Approximate point-to-surface distance using a dense reference cloud.
        from scipy.spatial import cKDTree

        # The cloud is intentionally dense relative to the default tolerance.
        ref_cloud, _ = _sample_surface_seeded(mesh, max(n_samples * 60, 100000),
                                              SYMMETRY_SAMPLE_SEED + 1)
        dist, _ = cKDTree(ref_cloud).query(reflected)
    rel = float(np.percentile(dist, 98)) / max(diag, 1e-9)
    return rel < rel_tol, rel


# =====================================================================
# 2. AST step splitter
# =====================================================================

#: CadQuery methods that *change observable 3D geometry* and therefore
#: terminate a step.  Selector / sketch-setup calls (faces, workplane,
#: circle, ...) are glued to the following terminal op.
TERMINAL_OPS: frozenset[str] = frozenset(
    {
        # primitives
        "box", "sphere", "cylinder", "wedge", "text",
        # 2D -> 3D
        "extrude", "twistExtrude", "revolve", "loft", "sweep",
        # boolean / removal
        "cut", "cutBlind", "cutThruAll", "cutEach",
        "hole", "cboreHole", "cskHole",
        "union", "intersect", "combine", "split",
        # local features
        "fillet", "chamfer", "shell",
        # rigid transforms (move the bbox)
        "translate", "rotate", "rotateAboutCenter", "mirror",
    }
)

_CONTROL_FLOW = (ast.For, ast.While, ast.If, ast.FunctionDef, ast.AsyncFunctionDef,
                 ast.With, ast.Try, ast.ClassDef)


@dataclass
class Step:
    """One verifiable unit of the generated program."""

    index: int                 # 1-based global step index
    label: str                 # terminal op name (or "stmt" in fallback mode)
    target: str                # variable holding the result after this step
    step_source: str           # the (possibly truncated) statement of this step
    prefix_source: str         # full cumulative program up to & incl. this step


def _chain_calls(node: ast.expr) -> list[tuple[str, ast.Call]] | None:
    """Decompose ``a.b(..).c(..)`` into [(name, call_node)] innermost→outermost.

    Returns None if the expression is not a pure attribute-call chain.
    """
    calls: list[tuple[str, ast.Call]] = []
    cur = node
    while isinstance(cur, ast.Call):
        fn = cur.func
        if isinstance(fn, ast.Attribute):
            calls.append((fn.attr, cur))
            cur = fn.value
        elif isinstance(fn, ast.Name):
            calls.append((fn.id, cur))
            cur = None  # type: ignore[assignment]
            break
        else:
            return None
    if cur is not None and not isinstance(cur, ast.Name):
        return None
    calls.reverse()
    return calls


def _chain_root_ok(value: ast.expr, known_vars: set[str]) -> bool:
    """Is this expression a call-chain rooted in cadquery or a known var?"""
    cur = value
    while isinstance(cur, ast.Call):
        cur = cur.func.value if isinstance(cur.func, ast.Attribute) else None
        if cur is None:
            return True  # rooted in a bare Name(...) constructor call
    if isinstance(cur, ast.Name):
        return cur.id in known_vars | {"cq", "cadquery"}
    if isinstance(cur, ast.Attribute):  # e.g. cq.Solid.makeTorus
        base = cur
        while isinstance(base, ast.Attribute):
            base = base.value
        return isinstance(base, ast.Name) and base.id in {"cq", "cadquery"}
    return False


def split_steps(code: str) -> list[Step]:
    """Split CadQuery source into cumulative executable prefixes.

    Strategy ladder:
      1. *chain mode* — split fluent chains at TERMINAL_OPS;
      2. *statement mode* — if top-level control flow (or no chain) is
         present, fall back to one step per top-level statement.
    """
    tree = ast.parse(code)
    if any(isinstance(s, _CONTROL_FLOW) for s in tree.body):
        return _split_statements(tree)

    steps: list[Step] = []
    emitted: list[str] = []          # completed source lines for the prefix
    known_vars: set[str] = set()

    for stmt in tree.body:
        is_chain = (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and _chain_root_ok(stmt.value, known_vars)
            and _chain_calls(stmt.value) is not None
        )
        if not is_chain:
            emitted.append(ast.unparse(stmt))
            continue

        target = stmt.targets[0].id  # type: ignore[union-attr]
        known_vars.add(target)
        calls = _chain_calls(stmt.value)  # type: ignore[arg-type]
        assert calls is not None
        if not calls:                      # Treat an empty chain as a plain statement.
            emitted.append(ast.unparse(stmt))
            continue

        terminal_idx = [i for i, (name, _) in enumerate(calls) if name in TERMINAL_OPS]
        if not terminal_idx:
            terminal_idx = [len(calls) - 1]          # no terminal: whole stmt
        terminal_idx[-1] = len(calls) - 1            # absorb trailing selectors

        for k in terminal_idx:
            name, call_node = calls[k]
            truncated = ast.Assign(
                targets=[ast.Name(id=target, ctx=ast.Store())],
                value=copy.deepcopy(call_node),
            )
            ast.fix_missing_locations(truncated)
            step_src = ast.unparse(truncated)
            steps.append(
                Step(
                    index=len(steps) + 1,
                    label=name,
                    target=target,
                    step_source=step_src,
                    prefix_source="\n".join([*emitted, step_src]),
                )
            )
        emitted.append(ast.unparse(stmt))

    return steps if steps else _split_statements(tree)


def _split_statements(tree: ast.Module) -> list[Step]:
    """Fallback: one step per top-level statement."""
    steps: list[Step] = []
    emitted: list[str] = []
    last_target = "result"
    for stmt in tree.body:
        src = ast.unparse(stmt)
        emitted.append(src)
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
            last_target = stmt.targets[0].id
        steps.append(
            Step(
                index=len(steps) + 1,
                label="stmt",
                target=last_target,
                step_source=src,
                prefix_source="\n".join(emitted),
            )
        )
    return steps


# =====================================================================
# 3. Sandbox execution
# =====================================================================


class StepTimeout(Exception):
    pass


# SIGALRM/setitimer are unavailable on Windows.
_HAS_SIGALRM = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")


@contextlib.contextmanager
def _time_limit(seconds: float):
    """Apply a SIGALRM timeout on the Unix main thread."""
    def handler(signum, frame):  # noqa: ARG001
        raise StepTimeout(f"step exceeded {seconds:.0f}s wall-clock limit")

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _exec_with_timeout(source: str, filename: str, ns: dict, seconds: float) -> None:
    """Execute source with a hard timeout when SIGALRM is available."""
    code = compile(source, filename, "exec")
    if _HAS_SIGALRM and threading.current_thread() is threading.main_thread():
        with _time_limit(seconds):
            exec(code, ns)
        return

    # OCCT objects are unsafe to move across threads. Without SIGALRM, execute
    # in the current thread and leave process-level isolation to batch callers.
    _ = seconds
    exec(code, ns)


@dataclass
class Snapshot:
    """Execution outcome of one cumulative prefix."""

    step: Step
    ok: bool
    props: GeoProps | None = None
    shape: cq.Shape | None = None
    error: str | None = None


def execute_steps(steps: list[Step], timeout: float = 30.0) -> list[Snapshot]:
    """Execute every cumulative prefix and snapshot its geometry.

    Execution stops at the first failing step: later prefixes contain the
    same code and would fail identically — the failure is thereby
    *localized* to that step.
    """
    snaps: list[Snapshot] = []
    for step in steps:
        ns: dict[str, Any] = {"cq": cq, "cadquery": cq, "math": math}
        try:
            _exec_with_timeout(step.prefix_source, f"<step-{step.index}>", ns, timeout)
        except Exception:
            snaps.append(Snapshot(step, ok=False, error=traceback.format_exc(limit=2)))
            break

        shape = _to_shape(ns.get(step.target))
        props = GeoProps.from_shape(shape) if shape is not None else None
        snaps.append(Snapshot(step, ok=True, props=props, shape=shape))
    return snaps


# =====================================================================
# 4. Assertion DSL
# =====================================================================

_ASSERT_RE = re.compile(r"@assert\s+(.*)")
_KV_RE = re.compile(r"^(\w+)\s*=\s*(.+)$")
_TOL_RE = re.compile(r"\s+tol\s*=\s*([0-9.]+)\s*(%?)\s*$")

_REL_NAMES = frozenset(
    {"volume", "area", "bbox", "bbox_vol", "vertex_count", "edge_count",
     "face_count", "solid_count", "through_holes", "genus"}
)


@dataclass(frozen=True)
class Assertion:
    raw: str                       # original text after "@assert"
    scope: int | None           # step index, or None for final
    kind: str                      # "kv" | "rel" | "unparseable"
    key: str = ""                  # for kv
    value: str = ""                # for kv (raw value string)
    tol: float | None = None    # for kv (absolute or fraction)
    tol_rel: bool = True           # tol given as percentage?
    expr: ast.Expression | None = None  # for rel (compiled, validated)
    error: str = ""                # for unparseable (why parsing failed)

    @property
    def scope_text(self) -> str:
        return "final" if self.scope is None else f"step={self.scope}"


def parse_assertions(text: str) -> list[Assertion]:
    """Extract and parse every ``@assert`` line from a CoT string.

    Never raises on malformed lines: they yield ``kind="unparseable"``
    assertions (reported as skipped by ``check``, never evaluated) — a
    model's formatting failure is data, not an infrastructure crash.
    """
    out: list[Assertion] = []
    for m in _ASSERT_RE.finditer(text):
        body = m.group(1).strip()
        try:
            out.append(_parse_one(body))
        except Exception as exc:  # noqa: BLE001  malformed assertion line
            out.append(Assertion(body, None, "unparseable",
                                 error=f"{type(exc).__name__}: {exc}"))
    return out


def _parse_one(body: str) -> Assertion:
    """Parse one assertion body (text after ``@assert``); raises on malformed."""
    scope: int | None = None
    first, _, rest = body.partition(" ")
    if first == "final":
        body = rest.strip()
    elif first.startswith("step="):
        scope = int(first.split("=", 1)[1])
        body = rest.strip()
    raw_body = body  # scope stripped; tol kept for display

    tol, tol_rel = None, True
    tm = _TOL_RE.search(body)
    if tm:
        tol = float(tm.group(1)) / (100.0 if tm.group(2) == "%" else 1.0)
        tol_rel = tm.group(2) == "%"
        body = body[: tm.start()].strip()

    kv = _KV_RE.match(body)
    if kv and kv.group(1) in _ALL_KEYS:
        return Assertion(raw_body, scope, "kv",
                         key=kv.group(1), value=kv.group(2).strip(),
                         tol=tol, tol_rel=tol_rel)
    return Assertion(raw_body, scope, "rel", expr=_compile_relational(body))


def _compile_relational(body: str) -> ast.Expression:
    """Validate a relational expression against a strict whitelist."""
    expr = ast.parse(body, mode="eval")
    for node in ast.walk(expr):
        if isinstance(node, ast.Name):
            if node.id not in _REL_NAMES:
                raise ValueError(f"name {node.id!r} not allowed in assertion {body!r}")
        elif isinstance(node, ast.Attribute):
            if node.attr not in {"x", "y", "z", "volume", "diagonal"}:
                raise ValueError(f"attribute .{node.attr} not allowed in {body!r}")
        elif not isinstance(
            node,
            (ast.Expression, ast.Compare, ast.BinOp, ast.UnaryOp, ast.Constant,
             ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub,
             ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.BoolOp,
             ast.And, ast.Or),
        ):
            raise ValueError(f"disallowed syntax {type(node).__name__} in {body!r}")
    return expr


# ---- per-key checkers -------------------------------------------------

def _close(measured: float, expected: float, tol: float | None, tol_rel: bool,
           default_rel: float = 0.05) -> bool:
    if tol is None:
        tol, tol_rel = default_rel, True
    bound = tol * max(abs(expected), 1e-9) if tol_rel else tol
    return abs(measured - expected) <= bound


# Accept scientific notation in numeric assertion values.
_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _parse_triple(s: str) -> tuple[float, float, float]:
    nums = [float(x) for x in _NUM_RE.findall(s)]
    if len(nums) != 3:
        raise ValueError(f"expected 3 numbers in {s!r}")
    return nums[0], nums[1], nums[2]


def _check_kv(a: Assertion, cur: GeoProps, prev: GeoProps | None,
              shape: cq.Shape | None) -> tuple[bool | None, str]:
    """Returns (passed | None=indeterminate, measured-detail)."""
    k, v = a.key, a.value

    if k == "volume":
        return _close(cur.volume, float(v), a.tol, a.tol_rel), f"volume={cur.volume:.6g}"
    if k == "area":
        return _close(cur.area, float(v), a.tol, a.tol_rel), f"area={cur.area:.6g}"

    if k in ("bbox", "bbox_sorted"):
        exp = _parse_triple(v)
        got = cur.bbox.sorted() if k == "bbox_sorted" else (cur.bbox.x, cur.bbox.y, cur.bbox.z)
        if k == "bbox_sorted":
            exp = tuple(sorted(exp))  # type: ignore[assignment]
        ok = all(_close(g, e, a.tol, a.tol_rel) for g, e in zip(got, exp, strict=True))
        return ok, f"{k}=({got[0]:.6g},{got[1]:.6g},{got[2]:.6g})"

    if k in ("face_count", "edge_count", "vertex_count", "solid_count"):
        got = getattr(cur, k)
        band = int(a.tol) if (a.tol is not None and not a.tol_rel) else 0
        return abs(got - int(v)) <= band, f"{k}={got}"

    if k == "through_holes":
        if cur.genus is None:
            return None, "genus indeterminate (non-manifold Euler count)"
        return cur.genus == int(v), f"through_holes={cur.genus}"

    if k in ("is_valid", "is_watertight"):
        got = getattr(cur, k)
        return got == (v.lower() == "true"), f"{k}={got}"

    if k == "delta_volume":
        if prev is None:
            return None, "no previous step to diff against"
        d = cur.volume - prev.volume
        # Use a relative threshold with an absolute floating-point floor.
        eps = max(DELTA_VOL_EPS_REL * prev.volume, DELTA_VOL_EPS_ABS)
        detail = f"delta_volume={d:+.6g}"
        if v == "increase":
            return d > eps, detail
        if v == "decrease":
            return d < -eps, detail
        if v == "zero":
            return abs(d) <= eps, detail
        return _close(d, float(v), a.tol, a.tol_rel, default_rel=0.10), detail

    if k == "delta_faces":
        if prev is None:
            return None, "no previous step to diff against"
        d = cur.face_count - prev.face_count
        detail = f"delta_faces={d:+d}"
        if v == "increase":
            return d > 0, detail
        if v == "decrease":
            return d < 0, detail
        return d == int(v), detail

    if k == "symmetry":
        if shape is None:
            return None, "no solid available for symmetry sampling"
        results = []
        for plane in (p.strip().upper() for p in v.split(",")):
            sym, rel = check_symmetry(shape, plane)
            results.append((plane, sym, rel))
        ok = all(s for _, s, _ in results)
        detail = ", ".join(f"{p}: dev={r:.4f}" for p, _, r in results)
        return ok, detail

    return None, f"unknown key {k!r}"


_ALL_KEYS = frozenset(
    {"volume", "area", "bbox", "bbox_sorted", "face_count", "edge_count",
     "vertex_count", "solid_count", "through_holes", "is_valid",
     "is_watertight", "delta_volume", "delta_faces", "symmetry"}
)


# =====================================================================
# 5. Checking & reporting
# =====================================================================


@dataclass
class CheckResult:
    assertion: Assertion
    passed: bool | None         # None => skipped / indeterminate
    detail: str
    step: Step | None = None    # implicated step (for localization)

    @property
    def mark(self) -> str:
        return {True: "[PASS]", False: "[FAIL]", None: "[SKIP]"}[self.passed]


@dataclass
class Report:
    results: list[CheckResult]
    snapshots: list[Snapshot]
    execution_error: Snapshot | None = None

    @property
    def ok(self) -> bool:
        return self.execution_error is None and all(
            r.passed is not False for r in self.results
        )

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.passed is False]

    def counts(self) -> tuple[int, int, int]:
        p = sum(1 for r in self.results if r.passed is True)
        f = sum(1 for r in self.results if r.passed is False)
        s = sum(1 for r in self.results if r.passed is None)
        return p, f, s

    def render(self) -> str:
        lines: list[str] = []
        if self.execution_error is not None:
            st = self.execution_error.step
            lines.append(f"EXECUTION FAILED at step {st.index} [{st.label}]")
            lines.append(f"    -> step code: {st.step_source}")
            lines.append("    -> " + self.execution_error.error.strip().splitlines()[-1])
            lines.append("")
        p, f, s = self.counts()
        lines.append(f"ASSERTION REPORT - {p} passed, {f} failed, {s} skipped")
        for r in self.results:
            lines.append(f"{r.mark} {r.assertion.scope_text:8s} "
                         f"{r.assertion.raw:42s} => {r.detail}")
            if r.passed is False and r.step is not None:
                lines.append(f"    -> step {r.step.index} [{r.step.label}] code: "
                             f"{r.step.step_source}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "execution_error": (
                None
                if self.execution_error is None
                else {"step": self.execution_error.step.index,
                      "error": self.execution_error.error}
            ),
            "results": [
                {"assertion": r.assertion.raw, "scope": r.assertion.scope_text,
                 "passed": r.passed, "detail": r.detail,
                 "step": r.step.index if r.step else None}
                for r in self.results
            ],
        }


def _rel_detail(expr: ast.Expression, ns: dict[str, Any]) -> str:
    """Format the measured values of every name used in a relational expr."""
    names = sorted({n.id for n in ast.walk(expr) if isinstance(n, ast.Name)})
    parts = []
    for name in names:
        v = ns.get(name)
        if isinstance(v, BBox):
            parts.append(f"{name}=({v.x:.6g},{v.y:.6g},{v.z:.6g})")
        elif isinstance(v, float):
            parts.append(f"{name}={v:.6g}")
        else:
            parts.append(f"{name}={v}")
    return ", ".join(parts)


def check(code: str, cot: str, timeout: float = 30.0) -> Report:
    """Split, execute, and verify assertions against each geometry state."""
    steps = split_steps(code)
    snaps = execute_steps(steps, timeout=timeout)
    assertions = parse_assertions(cot)

    exec_fail = next((s for s in snaps if not s.ok), None)
    by_index = {s.step.index: s for s in snaps if s.ok and s.props is not None}
    final = snaps[-1] if snaps and snaps[-1].ok and snaps[-1].props else None

    results: list[CheckResult] = []
    for a in assertions:
        if a.kind == "unparseable":
            results.append(CheckResult(a, None,
                                       f"unparseable assertion ({a.error})"))
            continue
        snap = final if a.scope is None else by_index.get(a.scope)
        if snap is None:
            reason = ("execution failed before this scope"
                      if exec_fail is not None else "no solid at this scope")
            results.append(CheckResult(a, None, reason))
            continue

        prev = by_index.get(snap.step.index - 1)
        if a.kind == "rel":
            try:
                ns = snap.props.namespace()
                ok = bool(eval(compile(a.expr, "<assert>", "eval"),  # noqa: S307
                               {"__builtins__": {}}, ns))
                results.append(CheckResult(a, ok, _rel_detail(a.expr, ns),
                                           step=None if ok else snap.step))
            except Exception as exc:  # pragma: no cover - defensive
                results.append(CheckResult(a, None, f"eval error: {exc}"))
        else:
            try:
                passed, detail = _check_kv(a, snap.props,
                                           prev.props if prev else None, snap.shape)
            except Exception as exc:  # noqa: BLE001  known key, malformed value
                passed, detail = None, f"malformed assertion value ({exc})"
            results.append(CheckResult(a, passed, detail,
                                       step=None if passed is not False else snap.step))

    return Report(results=results, snapshots=snaps, execution_error=exec_fail)


# =====================================================================
# 6. Repair prompt
# =====================================================================

_REPAIR_TEMPLATE = """\
The CadQuery code below violates {n} geometric assertion(s) made in its own \
design reasoning. Patch ONLY the implicated step(s); keep all verified steps \
unchanged.

## Current code
```python
{code}
```

## Violated assertions
{violations}

Return the corrected code for the implicated step(s) only.
"""


def build_repair_prompt(code: str, report: Report) -> str:
    """Turn a failing report into a targeted, step-localized repair prompt."""
    items: list[str] = []
    if report.execution_error is not None:
        st = report.execution_error.step
        err = report.execution_error.error.strip().splitlines()[-1]
        items.append(f"- EXECUTION ERROR at step {st.index} [{st.label}]\n"
                     f"  step code: `{st.step_source}`\n  error: {err}")
    for r in report.failures:
        loc = (f" at step {r.step.index} [{r.step.label}]\n"
               f"  step code: `{r.step.step_source}`") if r.step else ""
        items.append(f"- ASSERTION `{r.assertion.scope_text} {r.assertion.raw}` "
                     f"FAILED: measured {r.detail}{loc}")
    return _REPAIR_TEMPLATE.format(n=len(items), code=code.strip(),
                                   violations="\n".join(items))
