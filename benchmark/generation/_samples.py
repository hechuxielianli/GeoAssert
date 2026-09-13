"""Synthetic programs used by benchmark smoke checks."""

BUGGY_CODE = """\
import cadquery as cq
result = cq.Workplane("XY").box(100, 60, 40)
result = result.faces(">Z").workplane().circle(10).cutBlind(5)
result = result.edges("|Z").fillet(5)
"""

FIXED_CODE = BUGGY_CODE.replace("cutBlind(5)", "cutThruAll()")
