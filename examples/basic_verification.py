"""Minimal executable-assertion example."""

from geoassert import check

PROGRAM = """\
import cadquery as cq
result = cq.Workplane("XY").box(40, 30, 10)
result = result.faces(">Z").workplane().circle(4).cutThruAll()
"""

ASSERTIONS = """\
@assert step=1 bbox=(40,30,10) tol=1%
@assert step=1 volume=12000 tol=1%
@assert step=2 delta_volume=decrease
@assert final through_holes=1
@assert final is_valid=true
"""


def main() -> int:
    report = check(PROGRAM, ASSERTIONS)
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
