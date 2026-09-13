"""Sanity checks for the geometry metrics."""

import cadquery as cq
import numpy as np

from geoassert.metrics import chamfer_distance, point_f1, volume_iou


def _box(l, w, h):
    return cq.Workplane("XY").box(l, w, h).val()


def _box_with_hole(r=0.1):
    return (cq.Workplane("XY").box(0.6, 0.4, 0.3)
            .faces(">Z").workplane().circle(r).cutThruAll().val())


def setup_function(_):
    np.random.seed(0)  # Keep point sampling deterministic.


def test_cd_self_near_zero():
    g = _box(0.6, 0.4, 0.3)
    # Self-distance reflects sampling noise and should remain below a scale change.
    cd_self = chamfer_distance(_box(0.6, 0.4, 0.3), g)
    cd_diff = chamfer_distance(_box(0.72, 0.48, 0.36), g)
    assert cd_self < 0.1 and cd_self < cd_diff


def test_cd_monotonic_in_scale():
    g = _box(0.6, 0.4, 0.3)
    cd10 = chamfer_distance(_box(0.66, 0.44, 0.33), g)   # +10%
    cd20 = chamfer_distance(_box(0.72, 0.48, 0.36), g)   # +20%
    assert cd10 < cd20


def test_iou_self_near_one():
    g = _box(0.6, 0.4, 0.3)
    assert volume_iou(_box(0.6, 0.4, 0.3), g) > 0.99


def test_iou_in_range_and_hole_lowers():
    g = _box(0.6, 0.4, 0.3)
    iou = volume_iou(_box_with_hole(), g)
    assert 0.0 <= iou <= 1.0
    assert iou < 0.99


def test_f1_self_high():
    g = _box(0.6, 0.4, 0.3)
    assert point_f1(_box(0.6, 0.4, 0.3), g)["f1"] > 0.9
