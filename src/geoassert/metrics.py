"""Geometry metrics used by the experimental benchmark tools.

Distances are normalized using the reference shape's bounding-box center and
diagonal. This keeps translation and scale errors observable while making
values comparable across differently sized parts.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import trimesh
from scipy.spatial import cKDTree

DEFAULT_N = 2048


def code_seed(code: str, n: int = 0) -> int:
    """Derive a deterministic surface-sampling seed from source code."""
    h = int(hashlib.sha1(code.encode("utf-8")).hexdigest()[:8], 16)
    return (h ^ n) & 0x7FFFFFFF


def _bbox_center_diag(shape) -> tuple[np.ndarray, float]:
    bb = shape.BoundingBox()
    center = np.array([(bb.xmin + bb.xmax) / 2.0,
                       (bb.ymin + bb.ymax) / 2.0,
                       (bb.zmin + bb.zmax) / 2.0])
    diag = math.sqrt(bb.xlen ** 2 + bb.ylen ** 2 + bb.zlen ** 2)
    return center, max(diag, 1e-9)


def shape_to_mesh(shape) -> trimesh.Trimesh:
    """Convert a CadQuery/OCCT shape to a trimesh mesh."""
    _, diag = _bbox_center_diag(shape)
    verts, tris = shape.tessellate(max(diag / 1000.0, 1e-4))
    return trimesh.Trimesh(vertices=[(p.x, p.y, p.z) for p in verts],
                           faces=tris, process=True)


def sample_points(shape, n: int = DEFAULT_N, seed: int | None = None) -> np.ndarray:
    """Sample surface points, deterministically when ``seed`` is provided."""
    mesh = shape_to_mesh(shape)
    if seed is None:
        pts, _ = trimesh.sample.sample_surface(mesh, n)
    else:
        try:
            pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
        except TypeError:
            # Older trimesh releases lack a seed argument.
            state = np.random.get_state()
            np.random.seed(seed % (2**32))
            try:
                pts, _ = trimesh.sample.sample_surface(mesh, n)
            finally:
                np.random.set_state(state)
    return np.asarray(pts, dtype=float)


def _normalize_to(pts: np.ndarray, center: np.ndarray, diag: float) -> np.ndarray:
    return (pts - center) / diag


def chamfer_distance_points(pred_pts: np.ndarray, gold_pts: np.ndarray, *,
                            center: np.ndarray | None = None,
                            diag: float | None = None,
                            normalize: bool = True) -> float:
    """Compute symmetric Chamfer distance between two point clouds."""
    pa = np.asarray(pred_pts, dtype=float)
    pb = np.asarray(gold_pts, dtype=float)
    if normalize:
        if center is None or diag is None:
            raise ValueError("normalize=True requires gold center and diag")
        pa = _normalize_to(pa, np.asarray(center, dtype=float), float(diag))
        pb = _normalize_to(pb, np.asarray(center, dtype=float), float(diag))
    d_ab, _ = cKDTree(pb).query(pa)
    d_ba, _ = cKDTree(pa).query(pb)
    return float(d_ab.mean() + d_ba.mean())


def gold_reference(shape, n: int = DEFAULT_N, seed: int | None = None) -> dict:
    """Build a reusable point-cloud reference for a shape."""
    center, diag = _bbox_center_diag(shape)
    return {"points": sample_points(shape, n, seed=seed), "center": center, "diag": diag}


def chamfer_distance(pred_shape, gold_shape, n: int = DEFAULT_N,
                     normalize: bool = True, pred_seed: int | None = None,
                     gold_seed: int | None = None) -> float:
    """Return bidirectional mean Chamfer distance between two shapes."""
    pa = sample_points(pred_shape, n, seed=pred_seed)
    pb = sample_points(gold_shape, n, seed=gold_seed)
    if normalize:
        cg, diag = _bbox_center_diag(gold_shape)
        return chamfer_distance_points(pa, pb, center=cg, diag=diag)
    return chamfer_distance_points(pa, pb, normalize=False)


def point_f1(pred_shape, gold_shape, n: int = DEFAULT_N, tau: float = 0.05,
             normalize: bool = True) -> dict:
    """Return point precision, recall, and F1 at distance threshold ``tau``."""
    pa = sample_points(pred_shape, n)
    pb = sample_points(gold_shape, n)
    if normalize:
        cg, diag = _bbox_center_diag(gold_shape)
        pa = _normalize_to(pa, cg, diag)
        pb = _normalize_to(pb, cg, diag)
    d_ab, _ = cKDTree(pb).query(pa)   # pred → gold
    d_ba, _ = cKDTree(pa).query(pb)   # gold → pred
    precision = float((d_ab < tau).mean())
    recall = float((d_ba < tau).mean())
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def volume_iou(pred_shape, gold_shape) -> float:
    """Return exact solid-volume IoU using OCCT boolean operations.

    The shapes must use the same coordinate system. ``nan`` is returned when
    OCCT cannot evaluate a pathological boolean operation.
    """
    try:
        vp = pred_shape.Volume()
        vg = gold_shape.Volume()
        vi = pred_shape.intersect(gold_shape).Volume()
    except Exception:  # noqa: BLE001 - OCCT may reject pathological geometry.
        return float("nan")
    union = vp + vg - vi
    return vi / union if union > 1e-12 else 0.0
