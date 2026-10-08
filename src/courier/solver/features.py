"""Fixed-width, explainable features derived from an oracle SceneGraph."""

from __future__ import annotations

import math

import numpy as np

from courier.common.domain import Action, CostProfile, Scene
from .graph import OracleSolver, _turn_kind


# Multiple views of the same graph let the strategy model learn conditional
# preferences without hiding graph traversal inside a black-box classifier.
PROFILE_LIBRARY: tuple[tuple[str, CostProfile], ...] = (
    ("unit", CostProfile()),
    ("crowd_lex", CostProfile(crowded=0.01)),
    ("crowd_0.25", CostProfile(crowded=0.25)),
    ("crowd_0.5", CostProfile(crowded=0.5)),
    ("crowd_1", CostProfile(crowded=1.0)),
    ("crowd_2", CostProfile(crowded=2.0)),
    ("crowd_4", CostProfile(crowded=4.0)),
    ("crowd_8", CostProfile(crowded=8.0)),
    ("crowd_20", CostProfile(crowded=20.0)),
    ("crowd_100", CostProfile(crowded=100.0)),
    ("cover_lex", CostProfile(covered=-0.01)),
    ("cover_0.1", CostProfile(covered=-0.1)),
    ("cover_0.25", CostProfile(covered=-0.25)),
    ("cover_0.4", CostProfile(covered=-0.4)),
    ("cover_0.6", CostProfile(covered=-0.6)),
    ("cover_0.75", CostProfile(covered=-0.75)),
    ("cover_0.9", CostProfile(covered=-0.9)),
    ("turn_lex", CostProfile(turn=0.01, u_turn=0.02)),
    ("turn_0.1", CostProfile(turn=0.1, u_turn=0.2)),
    ("turn_0.25", CostProfile(turn=0.25, u_turn=0.5)),
    ("turn_0.5", CostProfile(turn=0.5, u_turn=1.0)),
    ("turn_1", CostProfile(turn=1.0, u_turn=2.0)),
    ("turn_2", CostProfile(turn=2.0, u_turn=4.0)),
    ("turn_3", CostProfile(turn=3.0, u_turn=6.0)),
    ("turn_5", CostProfile(turn=5.0, u_turn=10.0)),
    ("prefer_left", CostProfile(left_turn=-0.2, right_turn=0.2, u_turn=0.4)),
    ("prefer_right", CostProfile(left_turn=0.2, right_turn=-0.2, u_turn=0.4)),
    ("stairs_bonus", CostProfile(stairs=-0.25)),
    ("stairs_penalty", CostProfile(stairs=2.0)),
    ("crowd2_cover04", CostProfile(crowded=2.0, covered=-0.4)),
    ("crowd8_turn05", CostProfile(crowded=8.0, turn=0.5, u_turn=1.0)),
    ("cover06_turn05", CostProfile(covered=-0.6, turn=0.5, u_turn=1.0)),
    ("crowd2_prefer_right", CostProfile(crowded=2.0, left_turn=0.2, right_turn=-0.2, u_turn=0.4)),
    ("cover04_prefer_left", CostProfile(covered=-0.4, left_turn=-0.2, right_turn=0.2, u_turn=0.4)),
    # Reversing against the robot's heading is the main difference between robots: when they
    # leave the shortest path it is almost always by exactly two steps to avoid a U-turn.
    ("uturn_1", CostProfile(u_turn=1.0)),
    ("uturn_3", CostProfile(u_turn=3.0)),
    ("uturn_5", CostProfile(u_turn=5.0)),
    ("uturn_10", CostProfile(u_turn=10.0)),
    # Per-robot weights from a coordinate search on train (scripts/solver/fit_cost_weights.py).
    ("fit_r1", CostProfile(crowded=2.0, covered=0.25, u_turn=0.25, left_turn=0.1)),
    ("fit_r1_rain", CostProfile(crowded=2.0, covered=2.0, u_turn=0.25, left_turn=-0.5, right_turn=-0.25)),
    ("fit_r3_rain", CostProfile(crowded=0.75, covered=-0.4, u_turn=0.5, right_turn=0.1)),
    ("fit_r3_dry", CostProfile(crowded=2.0, covered=0.5, left_turn=-0.1)),
    ("fit_r5", CostProfile(crowded=0.75, covered=-0.1, turn=0.5, u_turn=5.0, right_turn=0.1)),
    ("fit_r6_urgent", CostProfile(crowded=0.1, u_turn=3.0, left_turn=0.1, right_turn=0.1)),
    ("fit_r6_calm", CostProfile(crowded=0.5, covered=-0.4, u_turn=3.0, left_turn=0.1, right_turn=0.1)),
    ("fit_r7_fragile", CostProfile(crowded=2.0, covered=-0.1, turn=0.25, u_turn=5.0, left_turn=-0.1, right_turn=0.25)),
    ("fit_r7_sturdy", CostProfile(crowded=0.75, covered=-0.1, u_turn=0.5, left_turn=0.1, right_turn=0.1)),
    ("fit_r4", CostProfile(crowded=0.5, covered=-0.1, turn=0.1, right_turn=0.1)),
)

LANDMARK_TYPES = ("library", "dorm", "sports", "clinic", "canteen", "parking", "lecture", "lab", "office", "gate")
REF_KINDS = ("north", "south", "west", "east", "near", "far", "north_most", "south_most", "west_most", "east_most", "anchor_near")


def _one_hot(value, choices) -> list[float]:
    return [float(value == choice) for choice in choices]


def feature_names() -> tuple[str, ...]:
    names: list[str] = [
        "grid_rows", "grid_cols", "node_density", "edge_count", "robot_r", "robot_c",
        "weather_rain", "urgent", "fragile", "has_via", "goal_candidate_count", "via_candidate_count",
    ]
    names += [f"heading_{i}" for i in range(4)]
    names += [f"goal_{kind}" for kind in LANDMARK_TYPES]
    names += [f"via_{kind}" for kind in LANDMARK_TYPES]
    names += [f"goal_ref_{kind}" for kind in REF_KINDS]
    names += [f"via_ref_{kind}" for kind in REF_KINDS]
    for action in range(4):
        names += [
            f"a{action}_legal", f"a{action}_normal", f"a{action}_crowded", f"a{action}_covered",
            f"a{action}_stairs", f"a{action}_straight", f"a{action}_left", f"a{action}_right",
            f"a{action}_uturn", f"a{action}_next_degree", f"a{action}_waypoint_manhattan",
        ]
    for profile_name, _ in PROFILE_LIBRARY:
        for action in range(4):
            names += [
                f"{profile_name}_a{action}_cost",
                f"{profile_name}_a{action}_regret",
                f"{profile_name}_a{action}_rank",
                f"{profile_name}_a{action}_best",
            ]
    return tuple(names)


def extract_scene_features(scene: Scene, *, legged: bool) -> np.ndarray:
    robot_id = 4 if legged else 0
    adjacency = OracleSolver.adjacency(scene, robot_id)
    immediate = {arc.action: arc for arc in adjacency[scene.robot_rc]}
    goal_candidates = scene.landmark_candidates(scene.mission.goal, scene.mission.goal_ref)
    via_candidates = (
        scene.landmark_candidates(scene.mission.via, scene.mission.via_ref)
        if scene.mission.via is not None
        else frozenset()
    )
    if not goal_candidates:
        raise ValueError(f"{scene.scene_id}: no goal candidates")
    if scene.mission.via is not None and not via_candidates:
        raise ValueError(f"{scene.scene_id}: no via candidates")
    waypoint_candidates = via_candidates or goal_candidates

    values: list[float] = [
        float(scene.grid_rows),
        float(scene.grid_cols),
        len(scene.nodes) / (scene.grid_rows * scene.grid_cols),
        float(len(scene.edges)),
        scene.robot_rc[0] / max(1, scene.grid_rows - 1),
        scene.robot_rc[1] / max(1, scene.grid_cols - 1),
        float(scene.weather == "rain"),
        float(scene.mission.urgent),
        float(scene.mission.fragile),
        float(scene.mission.via is not None),
        float(len(goal_candidates)),
        float(len(via_candidates)),
    ]
    values += _one_hot(int(scene.robot_heading), range(4))
    values += _one_hot(scene.mission.goal, LANDMARK_TYPES)
    values += _one_hot(scene.mission.via, LANDMARK_TYPES)
    values += _one_hot(scene.mission.goal_ref.kind if scene.mission.goal_ref else None, REF_KINDS)
    values += _one_hot(scene.mission.via_ref.kind if scene.mission.via_ref else None, REF_KINDS)

    for action in Action:
        arc = immediate.get(action)
        turn_kind = _turn_kind(scene.robot_heading, action)
        if arc is None:
            values += [0.0] * 11
            continue
        waypoint_distance = min(
            abs(arc.target[0] - target[0]) + abs(arc.target[1] - target[1])
            for target in waypoint_candidates
        )
        values += [
            1.0,
            float(arc.status == "normal"),
            float(arc.status == "crowded"),
            float(arc.status == "covered"),
            float(arc.stairs),
            float(turn_kind == "straight"),
            float(turn_kind == "left"),
            float(turn_kind == "right"),
            float(turn_kind == "u_turn"),
            float(len(adjacency[arc.target])),
            float(waypoint_distance),
        ]

    for _, profile in PROFILE_LIBRARY:
        scores = OracleSolver(profile).score_actions(scene, robot_id, adjacency=adjacency)
        minimum = min(scores.values())
        ordered_costs = sorted(set(scores.values()))
        for action in Action:
            cost = scores.get(action)
            if cost is None:
                values += [1000.0, 1000.0, 4.0, 0.0]
            else:
                rank = next(index for index, value in enumerate(ordered_costs) if math.isclose(cost, value))
                values += [cost, cost - minimum, float(rank), float(math.isclose(cost, minimum))]
    result = np.asarray(values, dtype=np.float32)
    expected = len(feature_names())
    if result.shape != (expected,):
        raise AssertionError(f"feature shape {result.shape} != {(expected,)}")
    return result


def build_feature_matrix(scenes: tuple[Scene, ...], *, legged: bool) -> np.ndarray:
    return np.stack([extract_scene_features(scene, legged=legged) for scene in scenes])
