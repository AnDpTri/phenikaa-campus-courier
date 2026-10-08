"""Ground parsed specs on the map: TargetSpec + landmarks -> common.Mission.

Rules, all checked against every goal_ref/via_ref in train and validation:
  * north/south/west/east pick the copy of the type with the extreme row/column;
  * near/far pick the copy with the smaller/larger straight-line distance to the
    anchor (the statement's "duong chim bay");
  * north_most/... pick the landmark of any type with the extreme row/column;
  * anchor_near picks the landmark of any type closest to the anchor in grid
    (Manhattan) distance, Euclidean as tie-break.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from courier.common.domain import Mission, NodeRC, SpatialRef

from .parser import ParsedMission, TargetSpec

Landmarks = Sequence[tuple[str, NodeRC]]

_DIRECTION_KEY = {
    "north": lambda rc: rc[0],
    "south": lambda rc: -rc[0],
    "west": lambda rc: rc[1],
    "east": lambda rc: -rc[1],
}


def _anchor_rc(anchor: str | None, landmarks: Landmarks) -> NodeRC | None:
    nodes = [rc for kind, rc in landmarks if kind == anchor]
    return nodes[0] if nodes else None


def resolve_target(spec: TargetSpec, landmarks: Landmarks) -> tuple[str, SpatialRef | None] | None:
    """Landmark type and SpatialRef for one spec, or None if the map cannot satisfy it."""
    if spec.ref is not None and spec.ref.endswith("_most"):
        if not landmarks:
            return None
        key = _DIRECTION_KEY[spec.ref.removesuffix("_most")]
        kind, rc = min(landmarks, key=lambda landmark: key(landmark[1]))
        return kind, SpatialRef(kind=spec.ref, rc=rc)

    if spec.ref == "anchor_near":
        anchor = _anchor_rc(spec.anchor, landmarks)
        others = [(kind, rc) for kind, rc in landmarks if rc != anchor]
        if anchor is None or not others:
            return None
        kind, rc = min(
            others,
            key=lambda lm: (abs(lm[1][0] - anchor[0]) + abs(lm[1][1] - anchor[1]), math.dist(lm[1], anchor)),
        )
        return kind, SpatialRef(kind="anchor_near", rc=rc, anchor=spec.anchor)

    if spec.type is None:
        return None
    copies = [rc for kind, rc in landmarks if kind == spec.type]
    if len(copies) < 2 or spec.ref is None:
        return spec.type, None
    if spec.ref in _DIRECTION_KEY:
        return spec.type, SpatialRef(kind=spec.ref, rc=min(copies, key=_DIRECTION_KEY[spec.ref]))
    anchor = _anchor_rc(spec.anchor, landmarks)
    if anchor is None:
        return spec.type, None
    pick = min if spec.ref == "near" else max
    return spec.type, SpatialRef(kind=spec.ref, rc=pick(copies, key=lambda rc: math.dist(rc, anchor)), anchor=spec.anchor)


def resolve(parsed: ParsedMission, landmarks: Landmarks) -> Mission:
    """Build the solver's Mission. Falls back to a type on the map rather than failing."""
    goal = resolve_target(parsed.goal, landmarks) if parsed.goal is not None else None
    if goal is None or not any(kind == goal[0] for kind, _ in landmarks):
        goal = (_fallback_type(parsed, landmarks), None)
    via = resolve_target(parsed.via, landmarks) if parsed.via is not None else None
    if via is not None and not any(kind == via[0] for kind, _ in landmarks):
        via = None
    return Mission(
        text=parsed.text,
        goal=goal[0],
        goal_ref=goal[1],
        via=via[0] if via else None,
        via_ref=via[1] if via else None,
        urgent=parsed.urgent,
        fragile=parsed.fragile,
    )


def _fallback_type(parsed: ParsedMission, landmarks: Landmarks) -> str:
    present = [kind for kind, _ in landmarks]
    if parsed.goal is not None and parsed.goal.type in present:
        return parsed.goal.type
    return present[0] if present else "gate"
