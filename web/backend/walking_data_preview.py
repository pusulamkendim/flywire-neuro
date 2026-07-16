"""Measured FlyBody walking snippets converted into web-ready maneuver clips."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import numpy as np
from scipy.signal import hilbert
from scipy.spatial.transform import Rotation as R, Slerp

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = (
    PROJECT_DIR / "data" / "simulation" / "datasets_walking-imitation"
    / "walking-dataset-small_female-only_snippets-100_min-len-0.5s_trk-files-0-9.hdf5"
)
CACHE_PATH = Path(__file__).resolve().parent / "walk_cache" / "walking_data_preview_v1.json"
NEUROMECHFLY_WALK_CACHE = Path(__file__).resolve().parent / "walk_cache" / "walk_5.0s.json"

PREVIEW_HZ = 30
DATA_TO_MM = 10.0
TRANSITION_SECONDS = 0.14
TRANSITION_FORWARD_MM = 1.2
GAIT_CYCLE_START = 60
GAIT_CYCLE_FRAMES = 5
# FlyBody qpos indices for the two alternating tripod leg groups.
TRIPOD_A = (40, 73, 84)  # left-front, right-middle, left-hind femur
TRIPOD_B = (51, 62, 95)  # right-front, left-middle, right-hind femur

ROUTES = {
    "straight-fast": "001",
    "straight-long": "014",
    "straight-steady": "042",
    "straight-wide": "083",
    "turn-left-soft": "026",
    "turn-left-medium": "019",
    "turn-left-sharp": "062",
    "turn-left-long": "069",
    "turn-right-soft": "053",
    "turn-right-medium": "036",
    "turn-right-sharp": "003",
    "turn-right-long": "050",
}

CHAIN_ROUTE_NAME = "maneuver-chain"
CHAIN_ORDER = [
    "straight-fast",
    "turn-left-soft",
    "straight-long",
    "turn-right-medium",
    "straight-steady",
    "turn-left-sharp",
    "straight-wide",
    "turn-right-soft",
    "turn-left-medium",
    "turn-right-sharp",
    "turn-left-long",
    "turn-right-long",
]
AVAILABLE_ROUTES = frozenset((*ROUTES, CHAIN_ROUTE_NAME))


def _interpolate_pose(start: list[float], end: list[float], progress: float) -> list[float]:
    first = np.asarray(start, dtype=np.float64).reshape(-1, 7)
    second = np.asarray(end, dtype=np.float64).reshape(-1, 7)
    pose = first.copy()
    pose[:, :3] = first[:, :3] * (1.0 - progress) + second[:, :3] * progress
    for index in range(len(pose)):
        rotations = R.from_quat(np.vstack((first[index, 3:7], second[index, 3:7])))
        pose[index, 3:7] = Slerp([0.0, 1.0], rotations)([progress])[0].as_quat()
    return np.round(pose.reshape(-1), 5).tolist()


def _pose_for_phase(gait_cycle: list[list[float]], phase_rad: float) -> list[float]:
    sample = (phase_rad / (2.0 * np.pi) % 1.0) * len(gait_cycle)
    lo = int(np.floor(sample)) % len(gait_cycle)
    hi = (lo + 1) % len(gait_cycle)
    return _interpolate_pose(gait_cycle[lo], gait_cycle[hi], float(sample - np.floor(sample)))


def _convert_clip(*, group, timestep, gait_cycle, gait_stride_mm) -> dict:
    root = np.asarray(group["root_qpos"], dtype=np.float64)
    joints = np.asarray(group["qpos"], dtype=np.float64)
    source_quat = root[:, [4, 5, 6, 3]]
    source_quat /= np.linalg.norm(source_quat, axis=1, keepdims=True)
    source_yaw = np.unwrap(R.from_quat(source_quat).as_euler("xyz")[:, 2])
    yaw_delta = source_yaw - source_yaw[0]

    cos_yaw, sin_yaw = np.cos(-source_yaw[0]), np.sin(-source_yaw[0])
    align = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    local_xy = (root[:, :2] - root[0, :2]) @ align.T
    local_path_mm = local_xy * DATA_TO_MM
    cumulative_path_mm = np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(np.diff(local_path_mm, axis=0), axis=1))
    ))

    # Derive the measured alternating-tripod phase from the six femur signals.
    tripod_signal = (
        joints[:, TRIPOD_A].sum(axis=1) - joints[:, TRIPOD_B].sum(axis=1)
    )
    tripod_signal -= np.mean(tripod_signal)
    measured_phase = np.unwrap(np.angle(hilbert(tripod_signal)))
    measured_phase -= measured_phase[0]

    emit_every = max(1, round(1.0 / (PREVIEW_HZ * timestep)))
    indices = np.unique(np.append(np.arange(0, len(root), emit_every), len(root) - 1))
    frames = []

    for frame_index in indices:
        retarget_phase = cumulative_path_mm[frame_index] / gait_stride_mm * 2.0 * np.pi
        gait_sample = (retarget_phase / (2.0 * np.pi)) % 1.0
        scaled_gait = gait_sample * len(gait_cycle)
        gait_lo = int(np.floor(scaled_gait)) % len(gait_cycle)
        gait_hi = (gait_lo + 1) % len(gait_cycle)
        gait_alpha = float(scaled_gait - np.floor(scaled_gait))
        poses = _interpolate_pose(gait_cycle[gait_lo], gait_cycle[gait_hi], gait_alpha)

        frames.append({
            "t_ms": round(float(frame_index * timestep * 1000.0), 2),
            "fly_pos": np.round([
                local_xy[frame_index, 0] * DATA_TO_MM,
                local_xy[frame_index, 1] * DATA_TO_MM,
                0.0,
            ], 4).tolist(),
            "heading_delta_rad": round(float(yaw_delta[frame_index]), 7),
            "poses": poses,
            "measured_gait_phase_rad": round(float(measured_phase[frame_index]), 5),
            "retarget_gait_phase_rad": round(float(retarget_phase), 5),
            "phase": "walking_data_preview",
            "walk_state": "WALKING",
        })

    displacement = local_path_mm[-1] - local_path_mm[0]
    path_length = float(np.linalg.norm(np.diff(local_path_mm, axis=0), axis=1).sum())
    duration_seconds = float((len(root) - 1) * timestep)
    measured_cycles = float((measured_phase[-1] - measured_phase[0]) / (2.0 * np.pi))
    return {
        "source_frames": len(root),
        "preview_frames": len(frames),
        "source_duration_ms": round(duration_seconds * 1000.0, 2),
        "displacement_mm": round(float(np.linalg.norm(displacement)), 3),
        "path_length_mm": round(path_length, 3),
        "turn_degrees": round(float(np.degrees(yaw_delta[-1])), 3),
        "mean_speed_mm_s": round(path_length / duration_seconds, 3),
        "measured_gait_hz": round(measured_cycles / duration_seconds, 3),
        "retarget_gait_hz": round(path_length / gait_stride_mm / duration_seconds, 3),
        "frames": frames,
    }


def build_preview_cache() -> dict:
    import h5py

    nmf_cache = json.loads(NEUROMECHFLY_WALK_CACHE.read_text())
    geom_names = nmf_cache["geom_names"]
    gait_cycle = [
        frame["poses"]
        for frame in nmf_cache["frames"][
            GAIT_CYCLE_START:GAIT_CYCLE_START + GAIT_CYCLE_FRAMES
        ]
    ]
    if len(gait_cycle) != GAIT_CYCLE_FRAMES:
        raise ValueError("NeuromechFly walk cache does not contain the requested gait cycle")
    gait_root_positions = np.asarray([
        frame["fly_pos"]
        for frame in nmf_cache["frames"][
            GAIT_CYCLE_START:GAIT_CYCLE_START + GAIT_CYCLE_FRAMES + 1
        ]
    ])
    gait_stride_mm = float(np.linalg.norm(
        np.diff(gait_root_positions[:, :2], axis=0), axis=1
    ).sum())

    previews = {}
    with h5py.File(DATASET_PATH, "r") as dataset:
        timestep = float(dataset["timestep_seconds"][()])
        for route_name, trajectory_id in ROUTES.items():
            converted = _convert_clip(
                group=dataset["trajectories"][trajectory_id],
                timestep=timestep,
                gait_cycle=gait_cycle,
                gait_stride_mm=gait_stride_mm,
            )
            previews[route_name] = {"trajectory_id": trajectory_id, **converted}

    payload = {
        "schema_version": "walking-data-preview-v1",
        "source": {
            "dataset": str(DATASET_PATH.relative_to(PROJECT_DIR)),
            "source_timestep_seconds": timestep,
            "preview_hz": PREVIEW_HZ,
            "source_position_units": "cm",
            "output_position_units": "mm",
            "position_scale": DATA_TO_MM,
            "visual_model": "neuromechfly",
            "motor_retarget": "root_odometry_locked_neuromechfly_gait_cycle",
            "neuromechfly_pose_cache": str(NEUROMECHFLY_WALK_CACHE.relative_to(PROJECT_DIR)),
            "neuromechfly_stride_mm": round(gait_stride_mm, 5),
            "measured_fields": ["root_xy", "root_yaw", "tripod_gait_phase_diagnostic"],
            "policy_used": False,
        },
        "geom_names": geom_names,
        "gait_cycle": gait_cycle,
        "previews": previews,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(payload, separators=(",", ":")))
    return payload


def load_preview_cache() -> dict:
    if not CACHE_PATH.exists():
        return build_preview_cache()
    return json.loads(CACHE_PATH.read_text())


def _scene_positions(frames: list[dict]) -> np.ndarray:
    return np.asarray([[frame["fly_pos"][0], 0.0, -frame["fly_pos"][1]] for frame in frames])


def _fly_position(scene_position: np.ndarray) -> list[float]:
    return [float(scene_position[0]), float(-scene_position[2]), 0.0]


class WalkingDataPreviewBridge:
    """Replay one measured walk clip or a finite 12-clip maneuver chain."""

    def __init__(self):
        self.queue = self._loop = self._thread = None
        self.running = False
        self._route_name = None

    def start(self, loop, queue, route_name: str):
        if route_name not in AVAILABLE_ROUTES:
            raise ValueError(f"Unknown walking preview route: {route_name}")
        self.queue, self._loop = queue, loop
        self._route_name = route_name
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
            print(f"[WalkingPreview] ERROR: {exc}", flush=True)
            traceback.print_exc()
            self.running = False
            self._emit({"event": "walk_end", "walking_preview_error": str(exc)})

    def _run(self):
        cached = load_preview_cache()
        routes = CHAIN_ORDER if self._route_name == CHAIN_ROUTE_NAME else [self._route_name]
        self._emit({
            "event": "walk_init",
            "geom_names": cached["geom_names"],
            "render_mode": "walking_data_preview",
            "preview": {
                "route": self._route_name,
                "trajectory_type": "measured_walking_snippets",
                "segment_count": len(routes),
                "source": cached["source"]["dataset"],
            },
        })

        position = np.zeros(3, dtype=np.float64)
        heading = 0.0
        gait_phase = 0.0
        gait_cycle = cached["gait_cycle"]
        gait_stride_mm = float(cached["source"]["neuromechfly_stride_mm"])
        output_index = 0
        previous_pose = None
        previous_group_heading = 0.0
        interval = 1.0 / PREVIEW_HZ

        for segment_index, route in enumerate(routes):
            clip = cached["previews"][route]
            frames = clip["frames"]
            if previous_pose is not None:
                transition_steps = max(2, round(TRANSITION_SECONDS * PREVIEW_HZ))
                start = position.copy()
                end = start + R.from_euler("y", heading).apply(
                    np.array([TRANSITION_FORWARD_MM, 0.0, 0.0])
                )
                transition_phase = TRANSITION_FORWARD_MM / gait_stride_mm * 2.0 * np.pi
                for transition_index in range(1, transition_steps + 1):
                    if not self.running:
                        break
                    alpha = transition_index / transition_steps
                    smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                    transition_heading = previous_group_heading + (
                        heading - previous_group_heading
                    ) * smooth
                    self._emit({
                        "event": "walk_frame",
                        "t_ms": round(output_index * interval * 1000.0, 2),
                        "fly_pos": np.round(_fly_position(start * (1.0 - smooth) + end * smooth), 4).tolist(),
                        "body_heading_rad": round(float(transition_heading), 7),
                        "poses": _pose_for_phase(
                            gait_cycle, gait_phase + transition_phase * smooth
                        ),
                        "global_gait_phase_rad": round(float(
                            gait_phase + transition_phase * smooth
                        ), 5),
                        "phase": "walking_transition",
                        "walk_state": "WALKING",
                        "segment_route": f"transition:{route}",
                        "segment_index": segment_index,
                        "is_transition": True,
                    })
                    output_index += 1
                    time.sleep(interval)
                if not self.running:
                    break
                position = end
                gait_phase += transition_phase

            segment_start = position.copy()
            segment_heading = heading
            local_positions = _scene_positions(frames)
            for frame_index, source_frame in enumerate(frames):
                if not self.running:
                    break
                position = segment_start + R.from_euler("y", segment_heading).apply(
                    local_positions[frame_index] - local_positions[0]
                )
                frame = {
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
                    "preview_route": self._route_name,
                    "segment_route": route,
                    "segment_index": segment_index,
                    "maneuver_source": "measured_hdf5",
                    "is_transition": False,
                }
                self._emit(frame)
                output_index += 1
                time.sleep(interval)
            if not self.running:
                break
            gait_phase += float(frames[-1]["retarget_gait_phase_rad"])
            heading += float(frames[-1]["heading_delta_rad"])
            previous_pose = frames[-1]["poses"]
            previous_group_heading = heading

        self.running = False
        self._emit({
            "event": "walk_end",
            "preview_route": self._route_name,
            "preserve_world_orientation": True,
        })


if __name__ == "__main__":
    data = build_preview_cache()
    for name, preview in data["previews"].items():
        print(
            f"{name}: #{preview['trajectory_id']} {preview['preview_frames']} frames, "
            f"{preview['displacement_mm']} mm, {preview['turn_degrees']} deg"
        )
