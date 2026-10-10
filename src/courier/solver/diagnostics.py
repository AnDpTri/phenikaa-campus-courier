"""Read-only graph diagnostics; never change strategy predictions."""

from collections import deque

from .graph import OracleSolver


def reachable(adjacency, starts):
    visited = set(starts)
    queue = deque(starts)
    while queue:
        node = queue.popleft()
        for arc in adjacency.get(node, ()):
            if arc.target not in visited:
                visited.add(arc.target)
                queue.append(arc.target)
    return visited


def fallback_details(scene, robot_id, action, reason):
    adjacency = OracleSolver.adjacency(scene, robot_id)
    legal = sorted({int(arc.action) for arc in adjacency.get(scene.robot_rc, ())})
    goals = scene.landmark_candidates(scene.mission.goal, scene.mission.goal_ref)
    vias = scene.landmark_candidates(scene.mission.via, scene.mission.via_ref) if scene.mission.via else frozenset()
    from_robot = reachable(adjacency, (scene.robot_rc,))
    reached_vias = vias & from_robot
    after_via = reachable(adjacency, reached_vias) if vias else from_robot
    route_actions = []
    if goals and (not scene.mission.via or vias):
        route_actions = sorted(int(a) for a in OracleSolver().score_actions(scene, robot_id, adjacency=adjacency))

    if not legal:
        category = "no_legal_outgoing"
    elif not goals:
        category = "no_goal_candidates"
    elif scene.mission.via and not vias:
        category = "no_via_candidates"
    elif vias and not reached_vias:
        category = "via_unreachable"
    elif not goals & after_via:
        category = "goal_unreachable_after_via" if vias else "goal_unreachable"
    elif not route_actions:
        category = "no_route_after_first_step"
    else:
        category = "other_feature_error"

    # Check whether removing direction restrictions alone would restore a route.
    undirected = {node: set() for node in adjacency}
    for source, arcs in adjacency.items():
        for arc in arcs:
            undirected[source].add(arc.target)
            undirected[arc.target].add(source)

    def flood(starts):
        found, todo = set(starts), deque(starts)
        while todo:
            for nxt in undirected.get(todo.popleft(), ()):
                if nxt not in found:
                    found.add(nxt)
                    todo.append(nxt)
        return found

    weak_start = flood((scene.robot_rc,))
    weak_vias = vias & weak_start
    weak_after = flood(weak_vias) if vias else weak_start
    weak_route = bool(goals & weak_after) and (not vias or bool(weak_vias))
    mode = "heading" if int(action) == int(scene.robot_heading) and int(action) in legal else "lowest_legal_action"
    if not legal:
        mode = "heading_without_legal_action"
    elif reason and "repaired:" in reason:
        mode = "repaired_" + reason.rsplit("repaired:", 1)[1]
    elif reason and reason.endswith("; greedy"):
        mode = "greedy"
    return {
        "robot_id": robot_id,
        "action": int(action),
        "raw_reason": reason,
        "category": category,
        "fallback_mode": mode,
        "legal_actions": legal,
        "unit_route_actions": route_actions,
        "goal_candidates": len(goals),
        "via_candidates": len(vias),
        "reachable_node_count": len(from_robot),
        "weak_undirected_route_exists": weak_route,
    }
