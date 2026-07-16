"""Persistent DN-driven body runtime assembled from measured motor data.

Unlike the finite behavior preview bridges, this bridge owns one world pose for
its entire lifetime. Descending-neuron rates select a controller, while the
controller changes only the pose generator; position, heading, gait phase and
flight momentum survive every transition until the user presses Stop.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from flight_data_preview import CHAIN_ORDER as FLIGHT_ROUTES
from flight_data_preview import load_preview_cache as load_flight_cache
from motor_state_resolver import MotorStateResolver
from walking_data_preview import ROUTES as WALK_ROUTES
from walking_data_preview import _interpolate_pose, _pose_for_phase
from walking_data_preview import _scene_positions as walking_scene_positions
from walking_data_preview import load_preview_cache as load_walking_cache


BACKEND_DIR = Path(__file__).resolve().parent
WORLD_PATH = BACKEND_DIR / "worlds" / "microhabitat_v1.json"
GROOM_CACHE_PATH = BACKEND_DIR / "walk_cache" / "groom_eye_clean_v2_3.0s.json"
FEED_CACHE_PATH = BACKEND_DIR / "walk_cache" / "feed_flybody_4.0s.json"
FLIGHT_POSTURE_PATH = BACKEND_DIR / "walk_cache" / "flight_leg_posture_v3_4.2s.json"
NEUTRAL_POSE_PATH = BACKEND_DIR / "walk_cache" / "walk_5.0s.json"

FRAME_HZ = 30.0
DT_MS = 1000.0 / FRAME_HZ
POSE_BLEND_FRAMES = 5
BOUNDARY_MARGIN_MM = 60.0
BOUNDARY_ALERT_MM = 105.0
ROW_SPACING_MM = 70.0
WAYPOINT_RADIUS_MM = 28.0
MIN_FLIGHT_ALTITUDE_MM = 6.0
CRUISE_ALTITUDE_MM = 8.0
MAX_FLIGHT_ALTITUDE_MM = 38.0
FLIGHT_SOURCE_STEP = 2  # 120 Hz source at 0.5x playback -> 60 Hz, shown at 30 Hz.
VERTICAL_MANEUVER_INTERVAL = 8

HORIZONTAL_FLIGHT_ROUTES = tuple(
    route for route in FLIGHT_ROUTES if route not in {"vertical-climb", "vertical-dive"}
)


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _scene_position(fly_pos: list[float]) -> np.ndarray:
    return np.asarray([fly_pos[0], fly_pos[2], -fly_pos[1]], dtype=np.float64)


def _fly_position(scene_pos: np.ndarray) -> list[float]:
    return [
        round(float(scene_pos[0]), 4),
        round(float(-scene_pos[2]), 4),
        round(float(scene_pos[1]), 4),
    ]


def _horizontal_heading(rotation: R, fallback: float = 0.0) -> float:
    forward = rotation.apply(np.asarray([1.0, 0.0, 0.0]))
    if float(np.hypot(forward[0], forward[2])) < 1e-6:
        return fallback
    return float(np.arctan2(-forward[2], forward[0]))


def _coverage_waypoints(width: float, depth: float) -> list[np.ndarray]:
    xmin, xmax = -width / 2 + BOUNDARY_MARGIN_MM, width / 2 - BOUNDARY_MARGIN_MM
    zmin, zmax = -depth / 2 + BOUNDARY_MARGIN_MM, depth / 2 - BOUNDARY_MARGIN_MM
    rows = np.arange(zmin, zmax + 0.1, ROW_SPACING_MM)
    points = [np.asarray([0.0, 0.0])]
    for index, z_value in enumerate(rows):
        pair = (xmin, xmax) if index % 2 == 0 else (xmax, xmin)
        points.extend(np.asarray([x_value, z_value]) for x_value in pair)
    return points


def _frames_for_phase(cache: dict, phase: str) -> list[dict]:
    frames = [frame for frame in cache["frames"] if frame.get("phase") == phase]
    if not frames:
        raise ValueError(f"Cache contains no {phase!r} frames")
    return frames


class DigitalLifeBridge:
    """Run one continuous measured-data body until explicitly stopped."""

    def __init__(self, sync_to_brain_time: bool = False):
        self.sync_to_brain_time = bool(sync_to_brain_time)
        self.queue = None
        self._loop = None
        self._thread = None
        self.running = False
        self._ready = threading.Event()
        self._startup_error = None
        self._brain_lock = threading.Lock()
        self._latest_brain = {
            "t_ms": 0.0,
            "dn": {},
            "active_stimuli": [],
            "behavior_mode": "idle",
        }

    def start(self, loop, queue):
        if self.running:
            return
        self.queue, self._loop = queue, loop
        self.running = True
        self._ready.clear()
        self._startup_error = None
        self._thread = threading.Thread(target=self._run_safe, daemon=True)
        self._thread.start()
        # Asset parsing (including SciPy rotations) must finish before the
        # brain thread starts importing PyTorch. Otherwise SciPy can observe a
        # partially initialized torch module while probing array backends.
        if not self._ready.wait(timeout=15):
            self.running = False
            raise RuntimeError("Digital Life motor assets did not initialize in time")
        if self._startup_error is not None:
            self.running = False
            raise RuntimeError("Digital Life motor assets failed to initialize") from self._startup_error

    def stop(self):
        self.running = False

    def set_brain_frame(self, frame: dict):
        """Receive the latest neural frame without coupling both thread rates."""
        with self._brain_lock:
            self._latest_brain = {
                "t_ms": float(frame.get("t_ms", 0.0)),
                "dn": dict(frame.get("dn", {})),
                "active_stimuli": list(frame.get("active_stimuli", [])),
                "behavior_mode": frame.get("behavior_mode", "idle"),
            }

    def _brain_snapshot(self) -> dict:
        with self._brain_lock:
            return {
                **self._latest_brain,
                "dn": dict(self._latest_brain["dn"]),
                "active_stimuli": list(self._latest_brain["active_stimuli"]),
            }

    def _emit(self, payload: dict):
        future = asyncio.run_coroutine_threadsafe(self.queue.put(payload), self._loop)
        future.result(timeout=5)

    def _run_safe(self):
        try:
            self._run()
        except Exception as exc:
            import traceback
            print(f"[DigitalLife] ERROR: {exc}", flush=True)
            traceback.print_exc()
            self.running = False
            self._startup_error = exc
            self._ready.set()
            try:
                self._emit({"event": "walk_end", "digital_life_error": str(exc)})
            except Exception:
                pass

    def _load_assets(self):
        world = json.loads(WORLD_PATH.read_text())
        self.width = float(world["arena"]["width_mm"])
        self.depth = float(world["arena"]["depth_mm"])
        self.waypoints = _coverage_waypoints(self.width, self.depth)
        self.waypoint_index = 1

        walking = load_walking_cache()
        self.geom_names = walking["geom_names"]
        self.gait_cycle = walking["gait_cycle"]
        self.gait_stride_mm = float(walking["source"]["neuromechfly_stride_mm"])
        self.walk_clips = {}
        for route, preview in walking["previews"].items():
            positions = walking_scene_positions(preview["frames"])
            self.walk_clips[route] = {
                "frames": preview["frames"],
                "positions": positions - positions[0],
                "turn_rad": float(preview["frames"][-1]["heading_delta_rad"]),
            }

        flight = load_flight_cache()
        self.flight_clips = {}
        for route in FLIGHT_ROUTES:
            frames = flight["previews"][route]["frames"]
            first = _scene_position(frames[0]["fly_pos"])
            rotations = R.from_quat(np.asarray([frame["body_quat"] for frame in frames]))
            self.flight_clips[route] = {
                "frames": frames,
                "positions": np.asarray([
                    _scene_position(frame["fly_pos"]) - first for frame in frames
                ]),
                "rotations": rotations,
                "end_yaw": _horizontal_heading(rotations[-1]),
            }

        groom = json.loads(GROOM_CACHE_PATH.read_text())
        self.groom_frames = _frames_for_phase(groom, "grooming")
        # The controller cache has ~7 entry frames and ~7 recovery frames.
        self.groom_entry_end = min(7, len(self.groom_frames) - 1)
        self.groom_recovery_start = max(
            self.groom_entry_end + 1, len(self.groom_frames) - 7
        )

        self.feed_cache = json.loads(FEED_CACHE_PATH.read_text())
        self.feed_frames = self.feed_cache["frames"]
        self.feed_open_start = next(
            index for index, frame in enumerate(self.feed_frames)
            if frame.get("phase") == "opening"
        )
        self.feed_active_start = next(
            index for index, frame in enumerate(self.feed_frames)
            if frame.get("phase") == "feeding"
        )
        self.feed_active_end = max(
            index for index, frame in enumerate(self.feed_frames)
            if frame.get("phase") == "feeding"
        )
        self.feed_retract_start = next(
            index for index, frame in enumerate(self.feed_frames)
            if frame.get("phase") == "retracting"
        )

        posture = json.loads(FLIGHT_POSTURE_PATH.read_text())
        if posture["geom_names"] != self.geom_names:
            raise ValueError("Flight posture and measured walk geometry orders differ")
        self.takeoff_frames = _frames_for_phase(posture, "takeoff")
        self.landing_frames = _frames_for_phase(posture, "landing")

        neutral = json.loads(NEUTRAL_POSE_PATH.read_text())
        if neutral["geom_names"] != self.geom_names:
            raise ValueError("Neutral and measured walk geometry orders differ")
        self.neutral_pose = neutral["frames"][0]["poses"]

    def _reset_runtime(self):
        self.resolver = MotorStateResolver()
        self.position = np.zeros(3, dtype=np.float64)
        self.heading = 0.0
        self.rotation = R.identity()
        self.last_velocity = np.zeros(3, dtype=np.float64)
        self.gait_phase = 0.0
        self.walk_segment = None
        self.flight_segment = None
        self.last_walk_route = None
        self.last_flight_route = None
        self.maneuver_index = 0
        self.vertical_route = "vertical-climb"
        self.active_controller = None
        self.controller_cursor = 0
        self.last_base_pose = list(self.neutral_pose)
        self.blend_from_pose = None
        self.blend_cursor = POSE_BLEND_FRAMES
        self.feed_mode = "inactive"
        self.feed_cursor = self.feed_open_start
        self.output_index = 0
        self.body_simulation_ms = 0.0
        self.last_brain_t_ms = 0.0
        self.motor_accumulator_ms = 0.0
        self.last_motion_frame = None
        self.started_at = time.perf_counter()

    def _run(self):
        self._load_assets()
        self._reset_runtime()
        self._ready.set()
        self._emit({
            "event": "walk_init",
            "geom_names": self.geom_names,
            "render_mode": "digital_life",
            "preview": {
                "route": "persistent-dn-driven-life",
                "trajectory_type": "continuous_measured_motor_runtime",
                "walking_source": "walking-imitation HDF5",
                "flight_source": "flight-imitation HDF5",
                "arena_mm": [self.width, self.depth],
                "timing_mode": (
                    "brain_time_sync" if self.sync_to_brain_time else "wall_time"
                ),
            },
        })

        next_tick = time.perf_counter()
        while self.running:
            brain = self._brain_snapshot()
            if self.sync_to_brain_time:
                frame = self._brain_timed_frame(brain)
            else:
                decision = self.resolver.update(brain["dn"], DT_MS)
                self.body_simulation_ms += DT_MS
                frame = self._next_frame(decision, brain)
                self.last_motion_frame = frame
            self._emit(frame)
            self.output_index += 1

            next_tick += 1.0 / FRAME_HZ
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()

        self._emit({
            "event": "walk_end",
            "preview_route": "persistent-dn-driven-life",
            "preserve_world_orientation": True,
        })

    def _brain_timed_frame(self, brain: dict) -> dict:
        """Advance one 30 Hz motor sample per 33.3 ms of computed LIF time.

        The body thread still publishes at 30 Hz wall time so the browser stays
        responsive, but repeated hold frames do not advance pose or odometry.
        This makes playback speed follow measured brain throughput rather than
        a machine-specific hard-coded slowdown such as 1/440.
        """
        brain_t_ms = max(0.0, float(brain.get("t_ms", 0.0)))
        brain_delta_ms = max(0.0, brain_t_ms - self.last_brain_t_ms)
        self.last_brain_t_ms = brain_t_ms
        self.motor_accumulator_ms += brain_delta_ms

        advanced_frame = None
        while self.motor_accumulator_ms + 1e-9 >= DT_MS:
            decision = self.resolver.update(brain["dn"], DT_MS)
            self.body_simulation_ms += DT_MS
            advanced_frame = self._next_frame(decision, brain)
            self.motor_accumulator_ms -= DT_MS

        if advanced_frame is not None:
            self.last_motion_frame = advanced_frame
            return advanced_frame

        if self.last_motion_frame is None:
            # Publish one neutral frame immediately so a newly loaded model is
            # grounded correctly before the first 33.3 brain milliseconds.
            decision = self.resolver.update(brain["dn"], 0.0)
            initial = self._next_frame(decision, brain)
            initial["motor_frame_advanced"] = False
            self.last_motion_frame = initial
            return initial

        return self._hold_frame(brain)

    def _hold_frame(self, brain: dict) -> dict:
        """Publish current state without advancing measured motor data."""
        held = {
            key: value
            for key, value in self.last_motion_frame.items()
            if key not in {
                "poses", "t_ms", "brain_t_ms", "body_simulation_ms",
                "behavior_mode", "active_stimuli", "dn", "motor_decision",
                "motor_frame_advanced",
            }
        }
        motor = dict(self.last_motion_frame.get("motor_decision", {}))
        motor["controller_changed"] = False
        held.update({
            "event": "walk_frame",
            "t_ms": round((time.perf_counter() - self.started_at) * 1000.0, 2),
            "brain_t_ms": round(float(brain.get("t_ms", 0.0)), 2),
            "body_simulation_ms": round(self.body_simulation_ms, 2),
            "timing_mode": "brain_time_sync",
            "behavior_mode": motor.get("behavior_intent", "idle"),
            "active_stimuli": brain.get("active_stimuli", []),
            "dn": brain.get("dn", {}),
            "motor_decision": motor,
            "motor_frame_advanced": False,
        })
        return held

    def _next_frame(self, decision, brain):
        requested = decision.controller

        # Feeding owns a detailed mouth overlay. Normal transitions finish its
        # retraction before locomotion resumes; escape closes it immediately so
        # the time-critical takeoff state is never skipped.
        if self.feed_mode != "inactive":
            if requested == "flight_takeoff":
                self._close_feed_overlay()
            elif self.feed_mode == "complete":
                self._close_feed_overlay()
            elif self.feed_mode == "active" and requested != "feed":
                self.feed_mode = "retracting"
                self.feed_cursor = self.feed_retract_start
            if self.feed_mode != "inactive":
                return self._feed_frame(decision, brain)

        if requested == "feed":
            self._open_feed_overlay()
            return self._feed_frame(decision, brain)

        controller_changed = requested != self.active_controller
        if controller_changed:
            self._enter_controller(requested)

        if requested == "walk":
            payload = self._walk_frame(decision)
        elif requested == "backward":
            payload = self._backward_frame(decision)
        elif requested == "groom":
            payload = self._groom_frame(recovering=False)
        elif requested == "groom_recover":
            payload = self._groom_frame(recovering=True)
        elif requested == "flight_takeoff":
            payload = self._takeoff_frame(decision)
        elif requested == "flight":
            payload = self._flight_frame(decision)
        elif requested == "flight_landing":
            payload = self._landing_frame(decision)
        else:
            payload = self._idle_frame()

        return self._decorate(
            payload, decision, brain, controller_changed=controller_changed,
            executed_controller=requested,
        )

    def _enter_controller(self, controller: str):
        self.active_controller = controller
        self.controller_cursor = 0
        self.blend_from_pose = list(self.last_base_pose)
        self.blend_cursor = 0

        if controller in {"walk", "backward"}:
            self.walk_segment = None
        if controller == "groom":
            self.controller_cursor = 0
        elif controller == "groom_recover":
            self.controller_cursor = self.groom_recovery_start
        elif controller == "flight_takeoff":
            self.flight_segment = None
            self.transition_start_position = self.position.copy()
            self.transition_start_rotation = self.rotation
            self.transition_start_heading = self.heading
            self.takeoff_target_altitude = max(
                CRUISE_ALTITUDE_MM, float(self.position[1]) + 5.0
            )
        elif controller == "flight":
            self.flight_segment = None
        elif controller == "flight_landing":
            self.flight_segment = None
            self.transition_start_position = self.position.copy()
            self.transition_start_rotation = self.rotation
            horizontal = self.last_velocity.copy()
            horizontal[1] = 0.0
            speed = float(np.linalg.norm(horizontal))
            if speed < 1e-6:
                horizontal = R.from_euler("y", self.heading).apply([1.0, 0.0, 0.0])
                speed = 0.0
            self.landing_direction = horizontal / max(float(np.linalg.norm(horizontal)), 1e-6)
            self.landing_distance = float(np.clip(speed * 7.5, 2.0, 22.0))
            level_heading = _horizontal_heading(self.rotation, self.heading)
            self.landing_rotation = R.from_euler("y", level_heading)
            self.landing_slerp = Slerp(
                [0.0, 1.0],
                R.from_quat(np.vstack((
                    self.transition_start_rotation.as_quat(),
                    self.landing_rotation.as_quat(),
                ))),
            )

    def _blend_pose(self, target: list[float]) -> list[float]:
        if self.blend_from_pose is not None and self.blend_cursor < POSE_BLEND_FRAMES:
            if np.allclose(self.blend_from_pose, target, rtol=0.0, atol=1e-8):
                pose = target
                self.blend_cursor = POSE_BLEND_FRAMES
            else:
                alpha = _smoothstep((self.blend_cursor + 1) / POSE_BLEND_FRAMES)
                pose = _interpolate_pose(self.blend_from_pose, target, alpha)
                self.blend_cursor += 1
        else:
            pose = target
        self.last_base_pose = list(pose)
        return pose

    def _idle_frame(self):
        return {
            "poses": self._blend_pose(self.neutral_pose),
            "phase": "standing",
            "flight_state": "GROUNDED",
            "body_heading_rad": self.heading,
            "segment_route": "standing",
            "maneuver_source": "neuromechfly_neutral_pose",
        }

    def _walk_frame(self, decision):
        if self.walk_segment is None:
            route = self._select_walk_route(decision.steering)
            clip = self.walk_clips[route]
            self.walk_segment = {
                "route": route,
                "clip": clip,
                "index": 0,
                "start": self.position.copy(),
                "heading": self.heading,
                "gait_phase": self.gait_phase,
            }

        segment = self.walk_segment
        clip = segment["clip"]
        index = segment["index"]
        source = clip["frames"][index]
        previous = self.position.copy()
        rotation = R.from_euler("y", segment["heading"])
        self.position = segment["start"] + rotation.apply(clip["positions"][index])
        self.position[1] = 0.0
        self.heading = _wrap_angle(segment["heading"] + source["heading_delta_rad"])
        self.rotation = R.from_euler("y", self.heading)
        self.last_velocity = self.position - previous
        self.gait_phase = segment["gait_phase"] + source["retarget_gait_phase_rad"]
        pose = _pose_for_phase(self.gait_cycle, self.gait_phase)

        if index >= len(clip["frames"]) - 1:
            self.last_walk_route = segment["route"]
            self.walk_segment = None
            self.maneuver_index += 1
        else:
            segment["index"] += 1

        return {
            "poses": self._blend_pose(pose),
            "phase": "walking_maneuver",
            "flight_state": "GROUNDED",
            "walk_state": "WALKING",
            "body_heading_rad": self.heading,
            "global_gait_phase_rad": round(self.gait_phase, 5),
            "segment_route": segment["route"],
            "maneuver_source": "measured_walking_hdf5",
        }

    def _backward_frame(self, decision):
        previous = self.position.copy()
        speed = 0.72
        turn = decision.steering * 0.035
        self.heading = _wrap_angle(self.heading + turn)
        direction = R.from_euler("y", self.heading).apply([1.0, 0.0, 0.0])
        self.position -= direction * speed
        self._clamp_horizontal_position()
        self.position[1] = 0.0
        self.last_velocity = self.position - previous
        self.gait_phase -= speed / self.gait_stride_mm * 2.0 * np.pi
        self.rotation = R.from_euler("y", self.heading)
        return {
            "poses": self._blend_pose(_pose_for_phase(self.gait_cycle, self.gait_phase)),
            "phase": "backward_walking",
            "flight_state": "GROUNDED",
            "walk_state": "BACKWARD",
            "body_heading_rad": self.heading,
            "global_gait_phase_rad": round(self.gait_phase, 5),
            "segment_route": "reverse-retargeted-gait",
            "maneuver_source": "measured_gait_reverse_odometry",
        }

    def _groom_frame(self, recovering: bool):
        if recovering:
            index = min(self.controller_cursor, len(self.groom_frames) - 1)
            self.controller_cursor += 1
            phase = "groom_recovery"
        else:
            index = self.controller_cursor
            self.controller_cursor += 1
            if self.controller_cursor >= self.groom_recovery_start:
                self.controller_cursor = self.groom_entry_end
            phase = "eye_grooming"
        target = self.groom_frames[index]["poses"]
        self.position[1] = 0.0
        self.last_velocity[:] = 0.0
        return {
            "poses": self._blend_pose(target),
            "phase": phase,
            "flight_state": "GROUNDED",
            "body_heading_rad": self.heading,
            "segment_route": phase,
            "maneuver_source": "cached_eye_clean_controller",
        }

    def _takeoff_frame(self, decision):
        alpha = float(np.clip(
            (decision.state_elapsed_ms + DT_MS) / MotorStateResolver.TAKEOFF_MS,
            0.0,
            1.0,
        ))
        smooth = _smoothstep(alpha)
        previous = self.position.copy()
        direction = R.from_euler("y", self.transition_start_heading).apply([1.0, 0.0, 0.0])
        self.position = self.transition_start_position + direction * 2.0 * smooth
        self.position[1] = (
            self.transition_start_position[1] * (1.0 - smooth)
            + self.takeoff_target_altitude * smooth
        )
        self._clamp_horizontal_position()
        self.heading = self.transition_start_heading
        self.rotation = R.from_euler("y", self.heading)
        self.last_velocity = self.position - previous
        index = min(self.controller_cursor, len(self.takeoff_frames) - 1)
        self.controller_cursor += 1
        return {
            "poses": self._blend_pose(self.takeoff_frames[index]["poses"]),
            "phase": "takeoff",
            "flight_state": "TAKEOFF",
            "body_quat": np.round(self.rotation.as_quat(), 7).tolist(),
            "segment_route": "takeoff",
            "maneuver_source": "cached_leg_retraction_takeoff",
        }

    def _flight_frame(self, decision):
        if self.flight_segment is None:
            route = self._select_flight_route(decision.steering)
            clip = self.flight_clips[route]
            self.flight_segment = {
                "route": route,
                "clip": clip,
                "index": 0,
                "start": self.position.copy(),
                "heading": self.heading,
                "rotation": R.from_euler("y", self.heading),
            }

        segment = self.flight_segment
        clip = segment["clip"]
        index = segment["index"]
        source = clip["frames"][index]
        previous = self.position.copy()
        self.position = segment["start"] + segment["rotation"].apply(
            clip["positions"][index]
        )
        self.rotation = segment["rotation"] * clip["rotations"][index]
        self.heading = _horizontal_heading(self.rotation, self.heading)
        self.last_velocity = self.position - previous

        next_index = index + FLIGHT_SOURCE_STEP
        if next_index >= len(clip["frames"]):
            self.last_flight_route = segment["route"]
            self.flight_segment = None
            self.maneuver_index += 1
        else:
            segment["index"] = next_index

        return {
            "poses": self._blend_pose(source["poses"]),
            "phase": "flight_maneuver",
            "flight_state": "FLYING",
            "body_quat": np.round(self.rotation.as_quat(), 7).tolist(),
            "segment_route": segment["route"],
            "maneuver_source": "measured_flight_hdf5",
        }

    def _landing_frame(self, decision):
        alpha = float(np.clip(
            (decision.state_elapsed_ms + DT_MS) / MotorStateResolver.LANDING_MS,
            0.0,
            1.0,
        ))
        smooth = _smoothstep(alpha)
        previous = self.position.copy()
        self.position = (
            self.transition_start_position
            + self.landing_direction * self.landing_distance * (1.0 - (1.0 - alpha) ** 2)
        )
        self.position[1] = max(0.0, self.transition_start_position[1] * (1.0 - smooth))
        self._clamp_horizontal_position()
        self.rotation = self.landing_slerp([smooth])[0]
        self.heading = _horizontal_heading(self.rotation, self.heading)
        self.last_velocity = self.position - previous
        index = min(self.controller_cursor, len(self.landing_frames) - 1)
        self.controller_cursor += 1
        return {
            "poses": self._blend_pose(self.landing_frames[index]["poses"]),
            "phase": "landing",
            "flight_state": "LANDING",
            "body_quat": np.round(self.rotation.as_quat(), 7).tolist(),
            "segment_route": "landing",
            "maneuver_source": "cached_leg_extension_landing",
            "is_landing": True,
        }

    def _open_feed_overlay(self):
        self.feed_mode = "active"
        self.feed_cursor = self.feed_open_start
        self.active_controller = "feed"
        self.last_velocity[:] = 0.0
        self._emit({
            "event": "walk_init",
            "geom_names": self.feed_cache["geom_names"],
            "render_mode": "proboscis_overlay",
            "controller": "feed",
        })

    def _close_feed_overlay(self):
        self.feed_mode = "inactive"
        self.active_controller = None
        self._emit({
            "event": "walk_init",
            "geom_names": self.geom_names,
            "render_mode": "digital_life",
            "controller": "feed_retracted",
        })

    def _feed_frame(self, decision, brain):
        index = min(self.feed_cursor, len(self.feed_frames) - 1)
        source = self.feed_frames[index]
        executed = "feed" if self.feed_mode == "active" else "feed_retract"

        if self.feed_mode == "active":
            self.feed_cursor += 1
            if self.feed_cursor > self.feed_active_end:
                self.feed_cursor = self.feed_active_start
        else:
            self.feed_cursor += 1
            if self.feed_cursor >= len(self.feed_frames):
                # Keep this final neutral-mouth frame in overlay mode. The
                # following tick switches back to the NeuromechFly pose stream.
                self.feed_mode = "complete"

        self.position[1] = 0.0
        self.last_velocity[:] = 0.0
        payload = {
            "poses": source["poses"],
            "phase": source.get("phase", executed),
            "flight_state": "GROUNDED",
            "body_heading_rad": self.heading,
            "segment_route": executed,
            "maneuver_source": "flybody_proboscis_overlay",
        }
        return self._decorate(
            payload,
            decision,
            brain,
            controller_changed=index in {self.feed_open_start, self.feed_retract_start},
            executed_controller=executed,
        )

    def _decorate(
        self, payload, decision, brain, *, controller_changed, executed_controller
    ):
        distance = self._distance_to_boundary(self.position)
        motor = decision.to_dict()
        motor.update({
            "controller_changed": bool(controller_changed),
            "executed_controller": executed_controller,
        })
        return {
            "event": "walk_frame",
            "t_ms": round((time.perf_counter() - self.started_at) * 1000.0, 2),
            "brain_t_ms": round(float(brain.get("t_ms", 0.0)), 2),
            "body_simulation_ms": round(self.body_simulation_ms, 2),
            "timing_mode": (
                "brain_time_sync" if self.sync_to_brain_time else "wall_time"
            ),
            "motor_frame_advanced": True,
            "fly_pos": _fly_position(self.position),
            "behavior_mode": decision.behavior_intent,
            "active_stimuli": brain.get("active_stimuli", []),
            "dn": brain.get("dn", {}),
            "motor_decision": motor,
            "body_driven": True,
            "preview_route": "persistent-dn-driven-life",
            "maneuver_index": self.maneuver_index,
            "waypoint_index": self.waypoint_index,
            "waypoint_count": len(self.waypoints) - 1,
            "boundary_avoidance": distance <= BOUNDARY_ALERT_MM,
            "distance_to_boundary_mm": round(distance, 2),
            **payload,
        }

    def _advance_waypoint(self):
        while float(np.linalg.norm(
            self.waypoints[self.waypoint_index] - self.position[[0, 2]]
        )) < WAYPOINT_RADIUS_MM:
            self.waypoint_index += 1
            if self.waypoint_index >= len(self.waypoints):
                self.waypoint_index = 1

    def _distance_to_boundary(self, position) -> float:
        return float(min(
            self.width / 2 - abs(position[0]),
            self.depth / 2 - abs(position[2]),
        ))

    def _clamp_horizontal_position(self):
        self.position[0] = float(np.clip(
            self.position[0], -self.width / 2 + 1.0, self.width / 2 - 1.0
        ))
        self.position[2] = float(np.clip(
            self.position[2], -self.depth / 2 + 1.0, self.depth / 2 - 1.0
        ))

    def _select_walk_route(self, steering: float) -> str:
        self._advance_waypoint()
        target = self.waypoints[self.waypoint_index]
        desired = np.arctan2(
            -(target[1] - self.position[2]), target[0] - self.position[0]
        ) + steering * 0.65
        distance_before = float(np.linalg.norm(target - self.position[[0, 2]]))
        rotation = R.from_euler("y", self.heading)
        best = None
        for order, route in enumerate(WALK_ROUTES):
            clip = self.walk_clips[route]
            path = self.position + rotation.apply(clip["positions"])
            endpoint = path[-1]
            clearance = min(
                self.width / 2 - float(np.abs(path[:, 0]).max()),
                self.depth / 2 - float(np.abs(path[:, 2]).max()),
            )
            final_heading = _wrap_angle(self.heading + clip["turn_rad"])
            score = (
                (float(np.linalg.norm(target - endpoint[[0, 2]])) - distance_before) * 2.2
                + abs(_wrap_angle(desired - final_heading)) * 28.0
                + max(0.0, BOUNDARY_MARGIN_MM - clearance) * 1000.0
                + (12.0 if route == self.last_walk_route else 0.0)
                + order * 1e-4
            )
            if best is None or score < best[0]:
                best = (score, route)
        return best[1]

    def _select_flight_route(self, steering: float) -> str:
        self._advance_waypoint()
        if self.maneuver_index and self.maneuver_index % VERTICAL_MANEUVER_INTERVAL == 0:
            route = self.vertical_route
            clip = self.flight_clips[route]
            path = self.position + R.from_euler("y", self.heading).apply(clip["positions"])
            if (
                float(path[:, 1].min()) >= MIN_FLIGHT_ALTITUDE_MM
                and float(path[:, 1].max()) <= MAX_FLIGHT_ALTITUDE_MM
            ):
                self.vertical_route = (
                    "vertical-dive" if route == "vertical-climb" else "vertical-climb"
                )
                return route

        target = self.waypoints[self.waypoint_index]
        desired = np.arctan2(
            -(target[1] - self.position[2]), target[0] - self.position[0]
        ) + steering * 0.8
        distance_before = float(np.linalg.norm(target - self.position[[0, 2]]))
        rotation = R.from_euler("y", self.heading)
        best = None
        for order, route in enumerate(HORIZONTAL_FLIGHT_ROUTES):
            clip = self.flight_clips[route]
            path = self.position + rotation.apply(clip["positions"])
            endpoint = path[-1]
            clearance = min(
                self.width / 2 - float(np.abs(path[:, 0]).max()),
                self.depth / 2 - float(np.abs(path[:, 2]).max()),
            )
            altitude_penalty = (
                max(0.0, MIN_FLIGHT_ALTITUDE_MM - float(path[:, 1].min()))
                + max(0.0, float(path[:, 1].max()) - MAX_FLIGHT_ALTITUDE_MM)
            )
            final_heading = _wrap_angle(self.heading + clip["end_yaw"])
            score = (
                (float(np.linalg.norm(target - endpoint[[0, 2]])) - distance_before) * 1.8
                + abs(_wrap_angle(desired - final_heading)) * 34.0
                + max(0.0, BOUNDARY_MARGIN_MM - clearance) * 1000.0
                + altitude_penalty * 1000.0
                + (20.0 if route == self.last_flight_route else 0.0)
                + order * 1e-4
            )
            if best is None or score < best[0]:
                best = (score, route)
        return best[1]
