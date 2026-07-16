"""Boundary-aware room coverage assembled from measured walking snippets."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from walking_data_preview import (
    PREVIEW_HZ,
    ROUTES,
    TRANSITION_FORWARD_MM,
    TRANSITION_SECONDS,
    _fly_position,
    _interpolate_pose,
    _pose_for_phase,
    _scene_positions,
    load_preview_cache,
)


WORLD_PATH = Path(__file__).resolve().parent / "worlds" / "microhabitat_v1.json"
BOUNDARY_MARGIN_MM = 60.0
BOUNDARY_ALERT_MM = 100.0
ROW_SPACING_MM = 70.0
WAYPOINT_RADIUS_MM = 28.0


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _coverage_waypoints(width: float, depth: float) -> list[np.ndarray]:
    xmin, xmax = -width / 2 + BOUNDARY_MARGIN_MM, width / 2 - BOUNDARY_MARGIN_MM
    zmin, zmax = -depth / 2 + BOUNDARY_MARGIN_MM, depth / 2 - BOUNDARY_MARGIN_MM
    rows = np.arange(zmin, zmax + 0.1, ROW_SPACING_MM)
    waypoints = [np.array([0.0, 0.0], dtype=np.float64)]
    for index, z in enumerate(rows):
        if index % 2 == 0:
            waypoints.extend((np.array([xmin, z]), np.array([xmax, z])))
        else:
            waypoints.extend((np.array([xmax, z]), np.array([xmin, z])))
    return waypoints


def _clip_geometry(preview: dict) -> dict:
    frames = preview["frames"]
    positions = _scene_positions(frames)
    return {
        "frames": frames,
        "positions": positions - positions[0],
        "turn_rad": float(frames[-1]["heading_delta_rad"]),
    }


class ExplorationWalkingBridge:
    """Continuously choose measured walk clips to cover the millimetre arena."""

    def __init__(self):
        self.queue = self._loop = self._thread = None
        self.running = False

    def start(self, loop, queue):
        self.queue, self._loop = queue, loop
        self.running = True
        self._thread = threading.Thread(target=self._run_safe, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _emit(self, payload):
        asyncio.run_coroutine_threadsafe(self.queue.put(payload), self._loop).result(timeout=5)

    def _run_safe(self):
        try:
            self._run()
        except Exception as exc:
            import traceback
            print(f"[ExploreWalking] ERROR: {exc}", flush=True)
            traceback.print_exc()
            self.running = False
            self._emit({"event": "walk_end", "walking_explore_error": str(exc)})

    def _run(self):
        world = json.loads(WORLD_PATH.read_text())
        width = float(world["arena"]["width_mm"])
        depth = float(world["arena"]["depth_mm"])
        waypoints = _coverage_waypoints(width, depth)
        cached = load_preview_cache()
        clips = {
            route: _clip_geometry(cached["previews"][route]) for route in ROUTES
        }
        self._emit({
            "event": "walk_init",
            "geom_names": cached["geom_names"],
            "render_mode": "walking_data_explore",
            "preview": {
                "route": "measured-walk-room-coverage",
                "trajectory_type": "boundary_aware_measured_walking_controller",
                "composition": "measured_straight_left_right_snippets_with_pose_transitions",
                "maneuver_count": len(clips),
                "arena_mm": [width, depth],
                "boundary_margin_mm": BOUNDARY_MARGIN_MM,
                "source": cached["source"]["dataset"],
            },
        })

        position = np.zeros(3, dtype=np.float64)
        heading = 0.0
        gait_phase = 0.0
        gait_cycle = cached["gait_cycle"]
        gait_stride_mm = float(cached["source"]["neuromechfly_stride_mm"])
        waypoint_index = 1
        output_index = 0
        maneuver_index = 0
        previous_route = None
        previous_pose = None
        previous_group_heading = 0.0
        interval = 1.0 / PREVIEW_HZ

        while self.running:
            waypoint_index = self._advance_waypoint(position, waypoints, waypoint_index)
            route = self._select_route(
                position=position,
                heading=heading,
                target=waypoints[waypoint_index],
                clips=clips,
                width=width,
                depth=depth,
                previous_route=previous_route,
            )
            clip = clips[route]

            if previous_pose is not None:
                position, gait_phase, output_index = self._stream_transition(
                    position=position,
                    previous_pose=previous_pose,
                    next_pose=clip["frames"][0]["poses"],
                    previous_group_heading=previous_group_heading,
                    next_group_heading=heading,
                    output_index=output_index,
                    maneuver_index=maneuver_index,
                    route=route,
                    waypoint_index=waypoint_index,
                    waypoint_count=len(waypoints) - 1,
                    width=width,
                    depth=depth,
                    gait_cycle=gait_cycle,
                    gait_stride_mm=gait_stride_mm,
                    gait_phase=gait_phase,
                )
                if not self.running:
                    break

            segment_heading = heading
            position, gait_phase, output_index = self._stream_clip(
                position=position,
                segment_heading=segment_heading,
                clip=clip,
                route=route,
                output_index=output_index,
                maneuver_index=maneuver_index,
                waypoint_index=waypoint_index,
                waypoint_count=len(waypoints) - 1,
                width=width,
                depth=depth,
                gait_cycle=gait_cycle,
                gait_phase=gait_phase,
            )
            if not self.running:
                break

            heading = _wrap_angle(heading + clip["turn_rad"])
            previous_group_heading = heading
            previous_pose = clip["frames"][-1]["poses"]
            previous_route = route
            maneuver_index += 1

        self.running = False
        self._emit({
            "event": "walk_end",
            "preview_route": "measured-walk-room-coverage",
            "preserve_world_orientation": True,
        })

    @staticmethod
    def _advance_waypoint(position, waypoints, waypoint_index):
        while float(np.linalg.norm(waypoints[waypoint_index] - position[[0, 2]])) < WAYPOINT_RADIUS_MM:
            waypoint_index += 1
            if waypoint_index >= len(waypoints):
                waypoint_index = 1
        return waypoint_index

    @staticmethod
    def _distance_to_boundary(position, width, depth):
        return float(min(width / 2 - abs(position[0]), depth / 2 - abs(position[2])))

    def _select_route(
        self, *, position, heading, target, clips, width, depth, previous_route
    ):
        desired_heading = np.arctan2(
            -(target[1] - position[2]), target[0] - position[0]
        )
        distance_before = float(np.linalg.norm(target - position[[0, 2]]))
        start_rotation = R.from_euler("y", heading)
        best = None

        for order, (route, clip) in enumerate(clips.items()):
            path = position + start_rotation.apply(clip["positions"])
            endpoint = path[-1]
            distance_after = float(np.linalg.norm(target - endpoint[[0, 2]]))
            final_heading = _wrap_angle(heading + clip["turn_rad"])
            heading_error = abs(_wrap_angle(desired_heading - final_heading))
            clearance = min(
                width / 2 - float(np.abs(path[:, 0]).max()),
                depth / 2 - float(np.abs(path[:, 2]).max()),
            )
            boundary_penalty = max(0.0, BOUNDARY_MARGIN_MM - clearance) * 1000.0
            repetition_penalty = 12.0 if route == previous_route else 0.0
            progress_cost = (distance_after - distance_before) * 2.2
            score = (
                progress_cost
                + heading_error * 28.0
                + boundary_penalty
                + repetition_penalty
                + order * 1e-4
            )
            if best is None or score < best[0]:
                best = (score, route)
        return best[1]

    def _metadata(self, *, position, waypoint_index, waypoint_count, width, depth):
        distance = self._distance_to_boundary(position, width, depth)
        return {
            "preview_route": "measured-walk-room-coverage",
            "waypoint_index": waypoint_index,
            "waypoint_count": waypoint_count,
            "boundary_avoidance": distance <= BOUNDARY_ALERT_MM,
            "distance_to_boundary_mm": round(distance, 2),
        }

    def _stream_transition(
        self,
        *,
        position,
        previous_pose,
        next_pose,
        previous_group_heading,
        next_group_heading,
        output_index,
        maneuver_index,
        route,
        waypoint_index,
        waypoint_count,
        width,
        depth,
        gait_cycle,
        gait_stride_mm,
        gait_phase,
    ):
        start = position.copy()
        end = start + R.from_euler("y", next_group_heading).apply(
            np.array([TRANSITION_FORWARD_MM, 0.0, 0.0])
        )
        heading_delta = _wrap_angle(next_group_heading - previous_group_heading)
        transition_phase = TRANSITION_FORWARD_MM / gait_stride_mm * 2.0 * np.pi
        steps = max(2, round(TRANSITION_SECONDS * PREVIEW_HZ))
        interval = 1.0 / PREVIEW_HZ
        for index in range(1, steps + 1):
            if not self.running:
                break
            alpha = index / steps
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            current = start * (1.0 - smooth) + end * smooth
            self._emit({
                "event": "walk_frame",
                "t_ms": round(output_index * interval * 1000.0, 2),
                "fly_pos": np.round(_fly_position(current), 4).tolist(),
                "body_heading_rad": round(float(
                    previous_group_heading + heading_delta * smooth
                ), 7),
                "poses": _pose_for_phase(
                    gait_cycle, gait_phase + transition_phase * smooth
                ),
                "global_gait_phase_rad": round(float(
                    gait_phase + transition_phase * smooth
                ), 5),
                "phase": "walking_transition",
                "walk_state": "WALKING",
                "segment_route": f"transition:{route}",
                "maneuver_index": maneuver_index,
                "maneuver_source": "synthetic_pose_blend",
                "is_transition": True,
                **self._metadata(
                    position=current,
                    waypoint_index=waypoint_index,
                    waypoint_count=waypoint_count,
                    width=width,
                    depth=depth,
                ),
            })
            position = current
            output_index += 1
            time.sleep(interval)
        return position, gait_phase + transition_phase, output_index

    def _stream_clip(
        self,
        *,
        position,
        segment_heading,
        clip,
        route,
        output_index,
        maneuver_index,
        waypoint_index,
        waypoint_count,
        width,
        depth,
        gait_cycle,
        gait_phase,
    ):
        start = position.copy()
        rotation = R.from_euler("y", segment_heading)
        interval = 1.0 / PREVIEW_HZ
        for frame_index, source_frame in enumerate(clip["frames"]):
            if not self.running:
                break
            position = start + rotation.apply(clip["positions"][frame_index])
            self._emit({
                **source_frame,
                "event": "walk_frame",
                "t_ms": round(output_index * interval * 1000.0, 2),
                "fly_pos": np.round(_fly_position(position), 4).tolist(),
                "body_heading_rad": round(float(
                    segment_heading + source_frame["heading_delta_rad"]
                ), 7),
                "poses": _pose_for_phase(
                    gait_cycle,
                    gait_phase + source_frame["retarget_gait_phase_rad"],
                ),
                "global_gait_phase_rad": round(float(
                    gait_phase + source_frame["retarget_gait_phase_rad"]
                ), 5),
                "phase": "walking_maneuver",
                "walk_state": "WALKING",
                "segment_route": route,
                "maneuver_index": maneuver_index,
                "maneuver_source": "measured_hdf5",
                "is_transition": False,
                **self._metadata(
                    position=position,
                    waypoint_index=waypoint_index,
                    waypoint_count=waypoint_count,
                    width=width,
                    depth=depth,
                ),
            })
            output_index += 1
            time.sleep(interval)
        return (
            position,
            gait_phase + float(clip["frames"][-1]["retarget_gait_phase_rad"]),
            output_index,
        )
