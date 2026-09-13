"""Tests for the executable assertion checker."""

import cadquery as cq
import pytest

from geoassert import (
    GeoProps,
    build_repair_prompt,
    check,
    check_symmetry,
    execute_steps,
    parse_assertions,
    split_steps,
)

# ---------------------------------------------------------------------
# Reference programs
# ---------------------------------------------------------------------

MULTI_STMT = """\
import cadquery as cq
result = cq.Workplane("XY").box(100, 60, 40)
result = result.faces(">Z").workplane().circle(10).cutThruAll()
result = result.edges("|Z").fillet(5)
"""

SINGLE_CHAIN = """\
import cadquery as cq
result = cq.Workplane("XY").box(100, 60, 40)\
.faces(">Z").workplane().circle(10).cutThruAll()\
.edges("|Z").fillet(5)
"""

WITH_LOOP = """\
import cadquery as cq
result = cq.Workplane("XY").box(100, 60, 40)
for x in (-25, 25):
    result = result.faces(">Z").workplane().center(x, 0).circle(5).cutThruAll()
"""

BUGGY_CUT_UP = """\
import cadquery as cq
result = cq.Workplane("XY").box(100, 60, 40)
result = result.faces(">Z").workplane().circle(10).cutBlind(5)
result = result.edges("|Z").fillet(5)
"""

BUGGY_FILLET = """\
import cadquery as cq
result = cq.Workplane("XY").box(100, 60, 40)
result = result.faces(">Z").workplane().circle(10).cutThruAll()
result = result.edges("|Z").fillet(35)
"""

GOOD_COT = """\
[Step 1] base: 100x60x40 box
  @assert step=1 bbox=(100,60,40) tol=2%
  @assert step=1 volume=240000 tol=2%
  @assert step=1 face_count=6
[Step 2] cut a D20 through hole in the top face
  @assert step=2 delta_volume=decrease
  @assert step=2 delta_faces=+1
  @assert step=2 through_holes=1
[Step 3] fillet the 4 vertical edges, r=5
  @assert step=3 delta_volume=decrease
  @assert final through_holes=1
  @assert final is_watertight=true
  @assert final volume < 0.95 * bbox_vol
  @assert final bbox.x > bbox.y
  @assert final symmetry=XZ,YZ
"""


# ---------------------------------------------------------------------
# 1. Splitter
# ---------------------------------------------------------------------

class TestSplitter:
    def test_multi_statement_chain(self):
        steps = split_steps(MULTI_STMT)
        assert [s.label for s in steps] == ["box", "cutThruAll", "fillet"]
        assert all(s.target == "result" for s in steps)
        # prefixes are cumulative and self-contained
        assert "import cadquery" in steps[0].prefix_source
        assert "cutThruAll" not in steps[0].prefix_source
        assert "fillet" in steps[2].prefix_source

    def test_single_long_chain_is_split(self):
        steps = split_steps(SINGLE_CHAIN)
        assert [s.label for s in steps] == ["box", "cutThruAll", "fillet"]
        # step-1 prefix must truncate the chain after box(...)
        assert steps[0].prefix_source.rstrip().endswith("box(100, 60, 40)")

    def test_selectors_glued_to_terminal(self):
        steps = split_steps(MULTI_STMT)
        assert ".faces('>Z')" in steps[1].step_source
        assert "cutThruAll" in steps[1].step_source

    def test_control_flow_falls_back_to_statements(self):
        steps = split_steps(WITH_LOOP)
        assert [s.label for s in steps] == ["stmt", "stmt", "stmt"]

    def test_prefixes_execute_identically_for_both_styles(self):
        v_multi = execute_steps(split_steps(MULTI_STMT))[-1].props.volume
        v_chain = execute_steps(split_steps(SINGLE_CHAIN))[-1].props.volume
        assert v_multi == pytest.approx(v_chain, rel=1e-9)


# ---------------------------------------------------------------------
# 2. Geometric properties
# ---------------------------------------------------------------------

class TestGeoProps:
    def test_box(self):
        p = GeoProps.from_shape(cq.Workplane("XY").box(100, 60, 40).findSolid())
        assert p.volume == pytest.approx(240000)
        assert (p.bbox.x, p.bbox.y, p.bbox.z) == pytest.approx((100, 60, 40))
        assert (p.vertex_count, p.edge_count, p.face_count) == (8, 12, 6)
        assert p.genus == 0 and p.is_valid and p.is_watertight

    @pytest.mark.parametrize(
        "builder,expected_genus",
        [
            (lambda: cq.Workplane("XY").box(100, 60, 40)
                .faces(">Z").workplane().circle(10).cutThruAll(), 1),
            (lambda: cq.Workplane("XY").box(100, 60, 40)
                .faces(">Z").workplane().circle(10).cutBlind(-5), 0),
            (lambda: cq.Workplane("XY").box(100, 60, 40).faces(">Z").workplane()
                .pushPoints([(-25, 0), (25, 0)]).circle(8).cutThruAll(), 2),
            (lambda: cq.Workplane("XY").sphere(20), 0),
            (lambda: cq.Workplane("XY").add(cq.Solid.makeTorus(30, 8)), 1),
        ],
        ids=["through-hole", "blind-hole", "two-holes", "sphere", "torus"],
    )
    def test_genus_counts_through_holes_only(self, builder, expected_genus):
        shape = builder().vals()[0] if not builder().solids().vals() \
            else builder().solids().vals()[0]
        assert GeoProps.from_shape(shape).genus == expected_genus

    def test_symmetry(self):
        box = cq.Workplane("XY").box(100, 60, 40).findSolid()
        for plane in ("XY", "XZ", "YZ"):
            sym, _ = check_symmetry(box, plane)
            assert sym, f"box must be symmetric across {plane}"
        lshape = (cq.Workplane("XY").box(100, 60, 40)
                  .faces(">Z").workplane().center(25, 0).rect(50, 60)
                  .cutBlind(-20).findSolid())
        sym_yz, _ = check_symmetry(lshape, "YZ")   # broken by the offset cut
        sym_xz, _ = check_symmetry(lshape, "XZ")   # still symmetric in Y
        assert not sym_yz and sym_xz


# ---------------------------------------------------------------------
# 3. Assertion DSL
# ---------------------------------------------------------------------

class TestAssertionDSL:
    def test_parse_scopes_and_tolerance(self):
        a = parse_assertions("@assert step=2 volume=1000 tol=10%")[0]
        assert (a.scope, a.key, a.value, a.tol, a.tol_rel) == (2, "volume", "1000", 0.10, True)

        b = parse_assertions("@assert final face_count=7 tol=1")[0]
        assert (b.scope, b.tol, b.tol_rel) == (None, 1.0, False)

        c = parse_assertions("@assert bbox=(100,60,40)")[0]   # scope defaults to final
        assert c.scope is None and c.key == "bbox"

    def test_parse_relational(self):
        a = parse_assertions("@assert final volume < 0.8 * bbox_vol")[0]
        assert a.kind == "rel"

    def test_relational_whitelist_rejects_injection(self):
        # One malformed assertion must not terminate a batch.
        inj = "@assert final __import__('os').system('id') == 0"
        a = parse_assertions(inj)[0]
        assert a.kind == "unparseable" and a.expr is None
        r = check(MULTI_STMT, inj).results[0]
        assert r.passed is None and "unparseable" in r.detail

    def test_malformed_assertions_never_raise(self):
        cot = (
            "@assert final is_symmetric=true\n"
            "@assert final volume ≈ 0.5\n"
            "@assert step=abc face_count=6\n"
            "@assert final volume=abc\n"
        )
        parsed = parse_assertions(cot)
        assert [a.kind for a in parsed] == ["unparseable"] * 3 + ["kv"]
        report = check(MULTI_STMT, cot)
        passed, failed, skipped = report.counts()
        assert (passed, failed, skipped) == (0, 0, 4), report.render()

    def test_bbox_accepts_scientific_notation(self):
        # Regression: parse scientific notation emitted by %.6g.
        report = check(MULTI_STMT, "@assert step=1 bbox=(1e2,6e1,4e1) tol=2%")
        passed, failed, skipped = report.counts()
        assert (passed, failed, skipped) == (1, 0, 0), report.render()

    def test_parse_many_lines(self):
        assert len(parse_assertions(GOOD_COT)) == 12


# ---------------------------------------------------------------------
# 4. End-to-end
# ---------------------------------------------------------------------

class TestEndToEnd:
    def test_good_code_passes_all(self):
        report = check(MULTI_STMT, GOOD_COT)
        passed, failed, skipped = report.counts()
        assert report.ok, report.render()
        assert (passed, failed, skipped) == (12, 0, 0)

    def test_buggy_cut_is_localized_to_step_2(self):
        report = check(BUGGY_CUT_UP, GOOD_COT)
        assert not report.ok
        failed = {r.assertion.raw for r in report.failures}
        # cutBlind(+5) cuts upward into air: no volume removed, no hole made
        assert any("delta_volume=decrease" in f for f in failed)
        assert any("through_holes=1" in f for f in failed)
        for r in report.failures:
            if "step=2" in r.assertion.scope_text:
                assert r.step is not None and r.step.index == 2
                assert "cutBlind(5)" in r.step.step_source

    def test_oversized_fillet_fails_exactly_at_step_3(self):
        report = check(BUGGY_FILLET, GOOD_COT)
        assert report.execution_error is not None
        assert report.execution_error.step.index == 3
        assert report.execution_error.step.label == "fillet"
        # steps 1-2 executed fine before the failure
        assert sum(1 for s in report.snapshots if s.ok) == 2
        # assertions scoped to steps 1-2 are still evaluated
        step12 = [r for r in report.results
                  if r.assertion.scope in (1, 2) and r.passed is True]
        assert len(step12) == 6

    def test_repair_prompt_contains_localization(self):
        report = check(BUGGY_CUT_UP, GOOD_COT)
        prompt = build_repair_prompt(BUGGY_CUT_UP, report)
        assert "cutBlind(5)" in prompt
        assert "delta_volume" in prompt
        assert "Patch ONLY the implicated step(s)" in prompt

    def test_report_dict_roundtrip(self):
        d = check(MULTI_STMT, GOOD_COT).to_dict()
        assert d["ok"] is True
        assert len(d["results"]) == 12

    def test_fallback_mode_still_checks_final_assertions(self):
        cot = "@assert final through_holes=2\n@assert final is_watertight=true"
        report = check(WITH_LOOP, cot)
        assert report.ok, report.render()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
