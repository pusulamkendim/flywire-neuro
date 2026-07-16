"""Continuous arena coverage assembled from measured flight maneuvers."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from flight_data_preview import (
    CHAIN_ORDER,
    PLAYBACK_RATE,
    PREVIEW_SAMPLE_HZ,
    TRANSITION_FORWARD_MM,
    TRANSITION_SOURCE_SECONDS,
    WING_BEAT_HZ,
    WING_PATTERN_PATH,
    _fly_position,
    _interpolate_pose_sequence,
    _landing_pose_sequence,
    _pose_with_wings,
    _scene_position,
    _tucked_pose,
    _wing_angles,
    load_preview_cache,
)


WORLD_PATH = Path(__file__).resolve().parent / "worlds" / "microhabitat_v1.json"
BOUNDARY_MARGIN_MM = 60.0
BOUNDARY_ALERT_MM = 105.0
ROW_SPACING_MM = 70.0
WAYPOINT_RADIUS_MM = 28.0
START_ALTITUDE_MM = 14.0
MIN_ALTITUDE_MM = 6.0
MAX_ALTITUDE_MM = 38.0
LANDING_SECONDS = 1.1
MAX_LANDING_TRAVEL_MM = 80.0
VERTICAL_MANEUVER_INTERVAL = 8

MANEUVER_ROUTES = tuple(
    route for route in CHAIN_ORDER if route not in {"vertical-climb", "vertical-dive"}
)


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _horizontal_heading(rotation: R, fallback: float = 0.0) -> float:
    """Heading of the animal's actual forward vector, independent of bank/pitch."""
    forward = rotation.apply(np.array([1.0, 0.0, 0.0]))
    horizontal_length = float(np.hypot(forward[0], forward[2]))
    if horizontal_length < 1e-6:
        return fallback
    return float(np.arctan2(-forward[2], forward[0]))


def _coverage_waypoints(width: float, depth: float) -> list[np.ndarray]:
    """Lawnmower path whose turns begin before the physical room boundary."""
    xmin, xmax = -width / 2 + BOUNDARY_MARGIN_MM, width / 2 - BOUNDARY_MARGIN_MM
    zmin, zmax = -depth / 2 + BOUNDARY_MARGIN_MM, depth / 2 - BOUNDARY_MARGIN_MM
    rows = np.arange(zmin, zmax + 0.1, ROW_SPACING_MM)
    points = [np.array([0.0, 0.0], dtype=np.float64)]
    for index, z in enumerate(rows):
        if index % 2 == 0:
            points.extend((np.array([xmin, z]), np.array([xmax, z])))
        else:
            points.extend((np.array([xmax, z]), np.array([xmin, z])))
    return points


def _clip_geometry(preview: dict) -> dict:
    frames = preview["frames"]
    first = _scene_position(frames[0]["fly_pos"])
    relative_positions = np.asarray([
        _scene_position(frame["fly_pos"]) - first for frame in frames
    ])
    rotations = R.from_quat(np.asarray([frame["body_quat"] for frame in frames]))
    yaw = _horizontal_heading(rotations[-1])
    return {
        "frames": frames,
        "relative_positions": relative_positions,
        "rotations": rotations,
        "end_yaw": float(yaw),
    }


class ExplorationFlightBridge:
    """Select and stream measured clips along a boundary-aware coverage path."""

    def __init__(self):
        self.queue = None
        self.running = False
        self._loop = None
        self._thread = None
        self._land_requested = threading.Event()
        self._last_position = None
        self._last_motion = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    def start(self, loop, queue):
        self.queue = queue
        self._loop = loop
        self.running = True
        self._land_requested.clear()
        self._last_position = None
        self._last_motion = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self._thread = threading.Thread(target=self._run_safe, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def request_land(self):
        self._land_requested.set()

    def _emit(self, data):
        future = asyncio.run_coroutine_threadsafe(self.queue.put(data), self._loop)
        future.result(timeout=5)

    def _record_motion(self, position):
        """Remember the last rendered displacement for a tangent-continuous landing."""
        current = np.asarray(position, dtype=np.float64)
        if self._last_position is not None:
            delta = current - self._last_position
            if float(np.linalg.norm(delta)) > 1e-6:
                self._last_motion = delta
        self._last_position = current.copy()

    def _run_safe(self):
        try:
            self._run()
        except Exception as exc:
            import traceback
            print(f"[ExploreFlight] ERROR: {exc}", flush=True)
            traceback.print_exc()
            self.running = False
            self._emit({"event": "walk_end", "explore_error": str(exc)})

    def _run(self):
        world = json.loads(WORLD_PATH.read_text())
        width = float(world["arena"]["width_mm"])
        depth = float(world["arena"]["depth_mm"])
        waypoints = _coverage_waypoints(width, depth)
        cached = load_preview_cache()
        clips = {
            route: _clip_geometry(cached["previews"][route])
            for route in CHAIN_ORDER
        }
        geom_names, base_pose = _tucked_pose()
        pattern = np.load(WING_PATTERN_PATH)

        self._emit({
            "event": "walk_init",
            "geom_names": geom_names,
            "render_mode": "flight_explore",
            "preview": {
                "route": "measured-room-coverage",
                "trajectory_type": "boundary_aware_measured_maneuver_controller",
                "playback_rate": PLAYBACK_RATE,
                "composition": "measured_saccade_evasion_clips_with_recovery_transitions",
                "maneuver_count": len(clips),
                "arena_mm": [width, depth],
                "boundary_margin_mm": BOUNDARY_MARGIN_MM,
            },
        })

        position = np.array([0.0, START_ALTITUDE_MM, 0.0], dtype=np.float64)
        self._last_position = position.copy()
        heading = 0.0
        rotation = R.identity()
        waypoint_index = 1
        output_index = 0
        maneuver_index = 0
        last_route = None
        next_vertical_route = "vertical-climb"

        while self.running and not self._land_requested.is_set():
            waypoint_index = self._advance_waypoint(position, waypoints, waypoint_index)
            target = waypoints[waypoint_index]
            boundary_distance = self._distance_to_boundary(position, width, depth)

            vertical_due = (
                maneuver_index > 0
                and maneuver_index % VERTICAL_MANEUVER_INTERVAL == 0
                and boundary_distance > BOUNDARY_ALERT_MM + 30.0
            )
            if vertical_due and self._vertical_route_is_safe(
                next_vertical_route, position, heading, clips
            ):
                route = next_vertical_route
                next_vertical_route = (
                    "vertical-dive" if route == "vertical-climb" else "vertical-climb"
                )
            else:
                route = self._select_route(
                    position=position,
                    heading=heading,
                    target=target,
                    clips=clips,
                    width=width,
                    depth=depth,
                    last_route=last_route,
                )

            position, rotation, output_index = self._stream_clip(
                route=route,
                clip=clips[route],
                position=position,
                heading=heading,
                geom_names=geom_names,
                waypoint_index=waypoint_index,
                waypoint_count=len(waypoints) - 1,
                width=width,
                depth=depth,
                output_index=output_index,
                maneuver_index=maneuver_index,
            )
            if not self.running or self._land_requested.is_set():
                break

            position, rotation, heading, output_index = self._stream_recovery(
                position=position,
                rotation=rotation,
                base_pose=base_pose,
                geom_names=geom_names,
                pattern=pattern,
                waypoint_index=waypoint_index,
                waypoint_count=len(waypoints) - 1,
                width=width,
                depth=depth,
                output_index=output_index,
                previous_route=route,
                maneuver_index=maneuver_index,
            )
            last_route = route
            maneuver_index += 1

        if self.running and self._land_requested.is_set():
            self._stream_landing(
                position, rotation, geom_names, output_index, maneuver_index
            )
        self.running = False
        self._emit({"event": "walk_end", "preview_route": "measured-room-coverage"})

    @staticmethod
    def _advance_waypoint(position, waypoints, waypoint_index):
        while True:
            distance = float(np.linalg.norm(waypoints[waypoint_index] - position[[0, 2]]))
            if distance >= WAYPOINT_RADIUS_MM:
                return waypoint_index
            waypoint_index += 1
            if waypoint_index >= len(waypoints):
                waypoint_index = 1

    @staticmethod
    def _distance_to_boundary(position, width, depth):
        return float(min(width / 2 - abs(position[0]), depth / 2 - abs(position[2])))

    @staticmethod
    def _vertical_route_is_safe(route, position, heading, clips):
        local = clips[route]["relative_positions"]
        global_path = position + R.from_euler("y", heading).apply(local)
        return bool(
            global_path[:, 1].min() >= MIN_ALTITUDE_MM
            and global_path[:, 1].max() <= MAX_ALTITUDE_MM
        )

    def _select_route(
        self, *, position, heading, target, clips, width, depth, last_route
    ):
        desired_heading = np.arctan2(
            -(target[1] - position[2]), target[0] - position[0]
        )
        distance_before = float(np.linalg.norm(target - position[[0, 2]]))
        start_rotation = R.from_euler("y", heading)
        best = None

        for order, route in enumerate(MANEUVER_ROUTES):
            clip = clips[route]
            global_path = position + start_rotation.apply(clip["relative_positions"])
            endpoint = global_path[-1]
            distance_after = float(np.linalg.norm(target - endpoint[[0, 2]]))
            final_heading = _wrap_angle(heading + clip["end_yaw"])
            heading_error = abs(_wrap_angle(desired_heading - final_heading))

            physical_clearance = min(
                width / 2 - float(np.abs(global_path[:, 0]).max()),
                depth / 2 - float(np.abs(global_path[:, 2]).max()),
            )
            altitude_min = float(global_path[:, 1].min())
            altitude_max = float(global_path[:, 1].max())
            unsafe_boundary = max(0.0, BOUNDARY_MARGIN_MM - physical_clearance)
            unsafe_altitude = (
                max(0.0, MIN_ALTITUDE_MM - altitude_min)
                + max(0.0, altitude_max - MAX_ALTITUDE_MM)
            )
            repetition = 20.0 if route == last_route else 0.0
            progress_cost = (distance_after - distance_before) * 1.8
            score = (
                progress_cost
                + heading_error * 34.0
                + unsafe_boundary * 1000.0
                + unsafe_altitude * 1000.0
                + repetition
                + order * 1e-4
            )
            if best is None or score < best[0]:
                best = (score, route)
        return best[1]

    def _frame_metadata(
        self, *, position, width, depth, waypoint_index, waypoint_count
    ):
        distance = self._distance_to_boundary(position, width, depth)
        return {
            "preview_route": "measured-room-coverage",
            "waypoint_index": waypoint_index,
            "waypoint_count": waypoint_count,
            "boundary_avoidance": distance <= BOUNDARY_ALERT_MM,
            "distance_to_boundary_mm": round(distance, 2),
        }

    def _stream_clip(
        self,
        *,
        route,
        clip,
        position,
        heading,
        geom_names,
        waypoint_index,
        waypoint_count,
        width,
        depth,
        output_index,
        maneuver_index,
    ):
        start = position.copy()
        start_rotation = R.from_euler("y", heading)
        last_position = start
        last_rotation = start_rotation
        interval = 1.0 / (PREVIEW_SAMPLE_HZ * PLAYBACK_RATE)

        for frame_index, source_frame in enumerate(clip["frames"]):
            if not self.running or self._land_requested.is_set():
                break
            last_position = start + start_rotation.apply(
                clip["relative_positions"][frame_index]
            )
            last_rotation = start_rotation * clip["rotations"][frame_index]
            frame = {
                **source_frame,
                "event": "walk_frame",
                "t_ms": round(output_index * interval * 1000.0, 2),
                "fly_pos": np.round(_fly_position(last_position), 4).tolist(),
                "body_quat": np.round(last_rotation.as_quat(), 7).tolist(),
                "phase": "exploration_maneuver",
                "flight_state": "FLYING",
                "segment_route": route,
                "maneuver_index": maneuver_index,
                "maneuver_source": "measured_hdf5",
                "is_transition": False,
                **self._frame_metadata(
                    position=last_position,
                    width=width,
                    depth=depth,
                    waypoint_index=waypoint_index,
                    waypoint_count=waypoint_count,
                ),
            }
            self._record_motion(last_position)
            self._emit(frame)
            output_index += 1
            time.sleep(interval)
        return last_position, last_rotation, output_index

    def _stream_recovery(
        self,
        *,
        position,
        rotation,
        base_pose,
        geom_names,
        pattern,
        waypoint_index,
        waypoint_count,
        width,
        depth,
        output_index,
        previous_route,
        maneuver_index,
    ):
        # Recover bank/pitch without changing the current visual flight
        # direction. Euler yaw is not stable for banked flight poses.
        yaw = _horizontal_heading(rotation)
        level_rotation = R.from_euler("y", yaw)
        start = position.copy()
        end = start + level_rotation.apply(
            np.array([TRANSITION_FORWARD_MM, 0.0, 0.0])
        )
        end[1] = float(np.clip(end[1], MIN_ALTITUDE_MM + 2.0, MAX_ALTITUDE_MM - 2.0))
        steps = max(2, round(TRANSITION_SOURCE_SECONDS * PREVIEW_SAMPLE_HZ))
        blend = Slerp(
            [0.0, 1.0],
            R.from_quat(np.vstack((rotation.as_quat(), level_rotation.as_quat()))),
        )
        interval = 1.0 / (PREVIEW_SAMPLE_HZ * PLAYBACK_RATE)

        for index in range(1, steps + 1):
            if not self.running or self._land_requested.is_set():
                break
            alpha = index / steps
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            current = start * (1.0 - smooth) + end * smooth
            current_rotation = blend([smooth])[0]
            source_seconds = output_index / PREVIEW_SAMPLE_HZ
            wing = _wing_angles(pattern, source_seconds)
            self._emit({
                "event": "walk_frame",
                "t_ms": round(output_index * interval * 1000.0, 2),
                "source_t_ms": round(source_seconds * 1000.0, 2),
                "fly_pos": np.round(_fly_position(current), 4).tolist(),
                "body_quat": np.round(current_rotation.as_quat(), 7).tolist(),
                "poses": _pose_with_wings(base_pose, geom_names, wing),
                "wing_phase": round(float((source_seconds * WING_BEAT_HZ) % 1.0), 5),
                "phase": "flight_recovery",
                "flight_state": "FLYING",
                "segment_route": f"recovery:{previous_route}",
                "maneuver_index": maneuver_index,
                "maneuver_source": "synthetic_recovery",
                "is_transition": True,
                **self._frame_metadata(
                    position=current,
                    width=width,
                    depth=depth,
                    waypoint_index=waypoint_index,
                    waypoint_count=waypoint_count,
                ),
            })
            self._record_motion(current)
            output_index += 1
            position = current
            rotation = current_rotation
            time.sleep(interval)
        return position, rotation, float(yaw), output_index

    def _stream_landing(
        self, position, start_rotation, geom_names, output_index, maneuver_index
    ):
        sequence = _landing_pose_sequence(geom_names)
        # Preserve the instantaneous forward direction at the moment Land is
        # requested; only bank and pitch are levelled during descent.
        yaw = _horizontal_heading(start_rotation)
        level_rotation = R.from_euler("y", yaw)
        blend = Slerp(
            [0.0, 1.0],
            R.from_quat(np.vstack((start_rotation.as_quat(), level_rotation.as_quat()))),
        )
        start = position.copy()
        # Continue along the instantaneous flight path rather than snapping to
        # the body's local +X axis when Land is pressed. Travel distance is
        # chosen so a quadratic ease-out begins near the previous frame speed.
        raw_motion = np.asarray(self._last_motion, dtype=np.float64).copy()
        motion = raw_motion.copy()
        motion[1] = 0.0
        speed_per_frame = float(np.linalg.norm(motion))
        if speed_per_frame > 1e-6:
            travel_direction = motion / speed_per_frame
        else:
            forward = start_rotation.apply(np.array([1.0, 0.0, 0.0]))
            forward[1] = 0.0
            travel_direction = forward / max(float(np.linalg.norm(forward)), 1e-6)
        steps = round(LANDING_SECONDS * 60)
        vertical_tangent = float(np.clip(
            raw_motion[1] * steps,
            -start[1],
            start[1],
        ))
        travel_distance = float(np.clip(
            speed_per_frame * steps / 2.0,
            0.0,
            MAX_LANDING_TRAVEL_MM,
        ))
        for index in range(1, steps + 1):
            if not self.running:
                return
            alpha = index / steps
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            horizontal_progress = 1.0 - (1.0 - alpha) ** 2
            current = start + travel_direction * travel_distance * horizontal_progress
            hermite_start = 2.0 * alpha ** 3 - 3.0 * alpha ** 2 + 1.0
            hermite_tangent = alpha ** 3 - 2.0 * alpha ** 2 + alpha
            current[1] = max(
                0.0,
                hermite_start * start[1] + hermite_tangent * vertical_tangent,
            )
            self._emit({
                "event": "walk_frame",
                "t_ms": round((output_index + index) / 60 * 1000.0, 2),
                "source_t_ms": round(index / 60 * 1000.0, 2),
                "fly_pos": np.round(_fly_position(current), 4).tolist(),
                "body_quat": np.round(blend([smooth])[0].as_quat(), 7).tolist(),
                "poses": _interpolate_pose_sequence(sequence, smooth),
                "phase": "landing",
                "flight_state": "LANDING" if index < steps else "GROUNDED",
                "preview_route": "measured-room-coverage",
                "segment_route": "landing",
                "maneuver_index": maneuver_index,
                "maneuver_source": "cached_landing_pose",
                "is_landing": True,
                "boundary_avoidance": False,
            })
            time.sleep(1.0 / 60)
