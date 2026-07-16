"""Flight Data Preview v1: real saccade/evasion CoM data as web replay.

This intentionally does not run the trained TensorFlow policy. It converts a
curated set of reference trajectories from FlyBody's HDF5 imitation dataset
into compact 120 Hz caches, combines them with the existing tucked-leg flight
pose, and adds the measured wing pattern as restrained render-only motion.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = (
    PROJECT_DIR / "data" / "simulation" / "datasets_flight-imitation"
    / "flight-dataset_saccade-evasion_augmented.hdf5"
)
WING_PATTERN_PATH = (
    PROJECT_DIR / "data" / "simulation" / "datasets_flight-imitation"
    / "wing_pattern_fmech.npy"
)
BASE_POSE_CACHE = Path(__file__).resolve().parent / "walk_cache" / "flight_leg_posture_v3_4.2s.json"
INITIAL_POSE_PATH = PROJECT_DIR / "web" / "frontend" / "assets" / "fly_pose.json"
PREVIEW_CACHE = Path(__file__).resolve().parent / "walk_cache" / "flight_data_preview_v1.json"

SOURCE_SAMPLE_HZ = 5000
PREVIEW_SAMPLE_HZ = 120
PLAYBACK_RATE = 0.5
ALTITUDE_MM = 8.0
MIN_CLEARANCE_MM = 3.0
WING_BEAT_HZ = 218.0
# The raw pattern drives FlyBody joints. NeuromechFly wing meshes have a
# different local pivot, so a restrained render scale prevents hinge drift.
WING_ROTATION_SCALE = 0.14

ROUTES = {
    "evasion": {"trajectory_id": "038", "expected_type": "evasion_original"},
    "saccade": {"trajectory_id": "201", "expected_type": "saccade_original"},
    "evasion-right-long": {"trajectory_id": "059", "expected_type": "evasion_original"},
    "evasion-left-long": {"trajectory_id": "057", "expected_type": "evasion_original"},
    "evasion-right-climb": {"trajectory_id": "149", "expected_type": "evasion_reflected"},
    "evasion-left-level": {"trajectory_id": "177", "expected_type": "evasion_reflected"},
    "saccade-left-wide": {"trajectory_id": "221", "expected_type": "saccade_original"},
    "saccade-right-wide": {"trajectory_id": "210", "expected_type": "saccade_original"},
    "saccade-left-fast": {"trajectory_id": "239", "expected_type": "saccade_reflected"},
    "saccade-right-fast": {"trajectory_id": "265", "expected_type": "saccade_reflected"},
    "vertical-climb": {"trajectory_id": "024", "expected_type": "evasion_original"},
    "vertical-dive": {"trajectory_id": "034", "expected_type": "evasion_original"},
}

CHAIN_ROUTE_NAME = "maneuver-chain"
CHAIN_ORDER = [
    "evasion",
    "saccade-left-wide",
    "evasion-right-long",
    "saccade-right-wide",
    "evasion-left-long",
    "saccade-left-fast",
    "evasion-right-climb",
    "vertical-climb",
    "saccade-right-fast",
    "evasion-left-level",
    "vertical-dive",
    "saccade",
]
AVAILABLE_ROUTES = frozenset((*ROUTES, CHAIN_ROUTE_NAME))
TRANSITION_SOURCE_SECONDS = 0.12
TRANSITION_FORWARD_MM = 3.0
LANDING_SOURCE_SECONDS = 0.7
LANDING_FORWARD_MM = 18.0


def _tucked_pose() -> tuple[list[str], np.ndarray]:
    if not BASE_POSE_CACHE.exists():
        raise FileNotFoundError(f"Flight posture cache is missing: {BASE_POSE_CACHE}")
    cached = json.loads(BASE_POSE_CACHE.read_text())
    frame = next((item for item in cached["frames"] if item.get("phase") == "flying"), None)
    if frame is None:
        raise ValueError("Flight posture cache contains no flying frame")
    geom_names = cached["geom_names"]
    pose = np.asarray(frame["poses"], dtype=np.float64).reshape(-1, 7)

    # The cached flying frame was captured at an arbitrary wingbeat phase.
    # Replace only the two wing transforms with the stable neutral reference;
    # the measured pattern is applied relative to these anchored transforms.
    initial = json.loads(INITIAL_POSE_PATH.read_text())
    initial_names = initial["geom_names"]
    initial_pose = np.asarray(initial["initial_pose"], dtype=np.float64).reshape(-1, 7)
    for wing_name in ("LWing", "RWing"):
        pose[geom_names.index(wing_name)] = initial_pose[initial_names.index(wing_name)]
    return geom_names, pose


def _wing_angles(pattern: np.ndarray, source_t: float) -> np.ndarray:
    phase = (source_t * WING_BEAT_HZ) % 1.0
    sample = phase * len(pattern)
    lo = int(np.floor(sample)) % len(pattern)
    hi = (lo + 1) % len(pattern)
    alpha = sample - np.floor(sample)
    angles = pattern[lo] * (1.0 - alpha) + pattern[hi] * alpha
    return np.concatenate((angles, angles))


def _pose_with_wings(
    base_pose: np.ndarray,
    geom_names: list[str],
    wing_angles: np.ndarray,
) -> list[float]:
    pose = base_pose.copy()
    for name, angles in (("LWing", wing_angles[:3]), ("RWing", wing_angles[3:])):
        if name not in geom_names:
            continue
        index = geom_names.index(name)
        base_rotation = R.from_quat(pose[index, 3:7])
        wing_delta = R.from_euler("yzx", angles * WING_ROTATION_SCALE)
        pose[index, 3:7] = (base_rotation * wing_delta).as_quat()
    return np.round(pose.reshape(-1), 5).tolist()


def _landing_pose_sequence(geom_names: list[str]) -> list[np.ndarray]:
    cached = json.loads(BASE_POSE_CACHE.read_text())
    if cached["geom_names"] != geom_names:
        raise ValueError("Landing cache geometry order does not match preview geometry")
    landing = [
        np.asarray(frame["poses"], dtype=np.float64).reshape(-1, 7)
        for frame in cached["frames"] if frame.get("phase") == "landing"
    ]
    ground = next(
        np.asarray(frame["poses"], dtype=np.float64).reshape(-1, 7)
        for frame in reversed(cached["frames"]) if frame.get("phase") == "walking"
    )
    if not landing:
        raise ValueError("Flight posture cache contains no landing frames")
    return [*landing, ground]


def _interpolate_pose_sequence(sequence: list[np.ndarray], progress: float) -> list[float]:
    scaled = np.clip(progress, 0.0, 1.0) * (len(sequence) - 1)
    lo = int(np.floor(scaled))
    hi = min(lo + 1, len(sequence) - 1)
    alpha = float(scaled - lo)
    pose = sequence[lo].copy()
    pose[:, :3] = sequence[lo][:, :3] * (1.0 - alpha) + sequence[hi][:, :3] * alpha
    if hi != lo:
        for index in range(len(pose)):
            rotations = R.from_quat(np.vstack((sequence[lo][index, 3:7], sequence[hi][index, 3:7])))
            pose[index, 3:7] = Slerp([0.0, 1.0], rotations)([alpha])[0].as_quat()
    return np.round(pose.reshape(-1), 5).tolist()


def _convert_route(group, geom_names: list[str], base_pose: np.ndarray, pattern: np.ndarray) -> dict:
    source_qpos = np.asarray(group["com_qpos"], dtype=np.float64)
    source_qvel = np.asarray(group["com_qvel"], dtype=np.float64)
    source_type = group["trajectory_type"][()]
    if isinstance(source_type, bytes):
        source_type = source_type.decode("utf-8")

    # Dataset quaternions are w,x,y,z; scipy/Three use x,y,z,w.
    source_quat = source_qpos[:, [4, 5, 6, 3]]
    source_quat /= np.linalg.norm(source_quat, axis=1, keepdims=True)
    source_t = np.arange(len(source_qpos), dtype=np.float64) / SOURCE_SAMPLE_HZ
    target_t = np.arange(0.0, source_t[-1] + 1e-12, 1.0 / PREVIEW_SAMPLE_HZ)

    relative_pos_cm = source_qpos[:, :3] - source_qpos[0, :3]
    position_cm = np.column_stack([
        np.interp(target_t, source_t, relative_pos_cm[:, axis]) for axis in range(3)
    ])

    source_rotation = R.from_quat(source_quat)
    relative_rotation = source_rotation * source_rotation[0].inv()
    conversion = R.from_euler("x", -90, degrees=True)
    scene_rotation = conversion * Slerp(source_t, relative_rotation)(target_t) * conversion.inv()

    frames = []
    for index, source_seconds in enumerate(target_t):
        pos_mm = position_cm[index] * 10.0
        pos_mm[2] += ALTITUDE_MM
        wing = _wing_angles(pattern, float(source_seconds))
        frames.append({
            "t_ms": round(float(source_seconds * 1000.0 / PLAYBACK_RATE), 2),
            "source_t_ms": round(float(source_seconds * 1000.0), 2),
            "fly_pos": np.round(pos_mm, 4).tolist(),
            "body_quat": np.round(scene_rotation[index].as_quat(), 7).tolist(),
            "poses": _pose_with_wings(base_pose, geom_names, wing),
            "wing_phase": round(float((source_seconds * WING_BEAT_HZ) % 1.0), 5),
            "phase": "flight_data_preview",
        })

    displacement_mm = float(np.linalg.norm(relative_pos_cm[-1]) * 10.0)
    path_mm = float(np.linalg.norm(np.diff(relative_pos_cm, axis=0), axis=1).sum() * 10.0)
    median_speed_mm_s = float(np.median(np.linalg.norm(source_qvel[:, :3], axis=1)) * 10.0)
    return {
        "trajectory_type": source_type,
        "source_frames": len(source_qpos),
        "source_duration_ms": round(float(source_t[-1] * 1000.0), 2),
        "preview_duration_ms": round(float(source_t[-1] * 1000.0 / PLAYBACK_RATE), 2),
        "displacement_mm": round(displacement_mm, 3),
        "path_length_mm": round(path_mm, 3),
        "median_speed_mm_s": round(median_speed_mm_s, 3),
        "frames": frames,
    }


def _scene_position(fly_pos: list[float]) -> np.ndarray:
    """MuJoCo-style [x,y,z] payload into Three.js [x,y,z]."""
    return np.array([fly_pos[0], fly_pos[2], -fly_pos[1]], dtype=np.float64)


def _fly_position(scene_pos: np.ndarray) -> list[float]:
    """Three.js [x,y,z] back into the existing fly_pos payload convention."""
    return [float(scene_pos[0]), float(-scene_pos[2]), float(scene_pos[1])]


def _chain_frame(
    *,
    chain_source_seconds: float,
    scene_pos: np.ndarray,
    scene_rotation: R,
    geom_names: list[str],
    base_pose: np.ndarray,
    pattern: np.ndarray,
    segment_index: int,
    segment_route: str,
    segment_source_ms: float,
    is_transition: bool,
    pose_override: list[float] | None = None,
    phase: str = "flight_data_preview",
    is_landing: bool = False,
) -> dict:
    wing = _wing_angles(pattern, chain_source_seconds)
    return {
        "t_ms": round(chain_source_seconds * 1000.0 / PLAYBACK_RATE, 2),
        "source_t_ms": round(chain_source_seconds * 1000.0, 2),
        "segment_source_t_ms": round(segment_source_ms, 2),
        "fly_pos": np.round(_fly_position(scene_pos), 4).tolist(),
        "body_quat": np.round(scene_rotation.as_quat(), 7).tolist(),
        "poses": pose_override or _pose_with_wings(base_pose, geom_names, wing),
        "wing_phase": round(float((chain_source_seconds * WING_BEAT_HZ) % 1.0), 5),
        "phase": phase,
        "segment_index": segment_index,
        "segment_route": segment_route,
        "is_transition": is_transition,
        "is_landing": is_landing,
        "flight_state": "LANDING" if is_landing else "FLYING",
    }


def _required_start_altitude(preview: dict) -> float:
    positions = np.asarray([
        _scene_position(frame["fly_pos"]) for frame in preview["frames"]
    ])
    relative_height = positions[:, 1] - positions[0, 1]
    return max(ALTITUDE_MM, MIN_CLEARANCE_MM - float(relative_height.min()))


def _build_maneuver_chain(
    previews: dict,
    geom_names: list[str],
    base_pose: np.ndarray,
    pattern: np.ndarray,
) -> dict:
    """Join measured clips with explicit synthetic recovery transitions."""
    frames = []
    chain_source_seconds = 0.0
    first_altitude = _required_start_altitude(previews[CHAIN_ORDER[0]])
    chain_pos = np.array([0.0, first_altitude, 0.0], dtype=np.float64)
    accumulated_yaw = R.identity()
    segment_manifest = []

    for segment_index, route_name in enumerate(CHAIN_ORDER):
        preview = previews[route_name]
        source_frames = preview["frames"]
        first_pos = _scene_position(source_frames[0]["fly_pos"])
        segment_start = chain_pos.copy()
        last_global_rotation = accumulated_yaw

        for frame_index, source_frame in enumerate(source_frames):
            # Adjacent clips share the transition endpoint; omit duplicate zero.
            if segment_index > 0 and frame_index == 0:
                continue
            local_pos = _scene_position(source_frame["fly_pos"]) - first_pos
            global_pos = segment_start + accumulated_yaw.apply(local_pos)
            local_rotation = R.from_quat(source_frame["body_quat"])
            global_rotation = accumulated_yaw * local_rotation
            segment_source_ms = float(source_frame["source_t_ms"])
            frames.append(_chain_frame(
                chain_source_seconds=chain_source_seconds,
                scene_pos=global_pos,
                scene_rotation=global_rotation,
                geom_names=geom_names,
                base_pose=base_pose,
                pattern=pattern,
                segment_index=segment_index,
                segment_route=route_name,
                segment_source_ms=segment_source_ms,
                is_transition=False,
            ))
            chain_source_seconds += 1.0 / PREVIEW_SAMPLE_HZ
            chain_pos = global_pos
            last_global_rotation = global_rotation

        segment_manifest.append({
            "index": segment_index,
            "route": route_name,
            "trajectory_id": preview["trajectory_id"],
            "trajectory_type": preview["trajectory_type"],
            "source_duration_ms": preview["source_duration_ms"],
        })

        if segment_index == len(CHAIN_ORDER) - 1:
            continue

        # Keep accumulated heading, but explicitly recover bank/pitch and
        # altitude between independently recorded clips.
        yaw, _, _ = last_global_rotation.as_euler("YXZ")
        next_yaw = R.from_euler("y", yaw)
        transition_start = chain_pos.copy()
        transition_end = transition_start + next_yaw.apply(
            np.array([TRANSITION_FORWARD_MM, 0.0, 0.0])
        )
        next_route = CHAIN_ORDER[segment_index + 1]
        transition_end[1] = _required_start_altitude(previews[next_route])
        transition_steps = round(TRANSITION_SOURCE_SECONDS * PREVIEW_SAMPLE_HZ)
        blend = Slerp(
            [0.0, 1.0],
            R.from_quat(np.vstack((last_global_rotation.as_quat(), next_yaw.as_quat()))),
        )
        for transition_index in range(1, transition_steps + 1):
            alpha = transition_index / transition_steps
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            transition_pos = transition_start * (1.0 - smooth) + transition_end * smooth
            transition_rotation = blend([smooth])[0]
            frames.append(_chain_frame(
                chain_source_seconds=chain_source_seconds,
                scene_pos=transition_pos,
                scene_rotation=transition_rotation,
                geom_names=geom_names,
                base_pose=base_pose,
                pattern=pattern,
                segment_index=segment_index,
                segment_route=f"transition:{route_name}",
                segment_source_ms=transition_index / PREVIEW_SAMPLE_HZ * 1000.0,
                is_transition=True,
            ))
            chain_source_seconds += 1.0 / PREVIEW_SAMPLE_HZ
        chain_pos = transition_end
        accumulated_yaw = next_yaw

    # The measured clips are airborne snippets. Finish the composed preview
    # with the existing leg-extension landing cache and an explicit ground
    # contact frame instead of leaving the animal suspended.
    landing_sequence = _landing_pose_sequence(geom_names)
    yaw, _, _ = last_global_rotation.as_euler("YXZ")
    landing_rotation = R.from_euler("y", yaw)
    landing_start = chain_pos.copy()
    landing_end = landing_start + landing_rotation.apply(
        np.array([LANDING_FORWARD_MM, 0.0, 0.0])
    )
    landing_end[1] = 0.0
    landing_steps = round(LANDING_SOURCE_SECONDS * PREVIEW_SAMPLE_HZ)
    landing_blend = Slerp(
        [0.0, 1.0],
        R.from_quat(np.vstack((last_global_rotation.as_quat(), landing_rotation.as_quat()))),
    )
    for landing_index in range(1, landing_steps + 1):
        alpha = landing_index / landing_steps
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        landing_pos = landing_start * (1.0 - smooth) + landing_end * smooth
        landing_frame = _chain_frame(
            chain_source_seconds=chain_source_seconds,
            scene_pos=landing_pos,
            scene_rotation=landing_blend([smooth])[0],
            geom_names=geom_names,
            base_pose=base_pose,
            pattern=pattern,
            segment_index=len(CHAIN_ORDER),
            segment_route="landing",
            segment_source_ms=landing_index / PREVIEW_SAMPLE_HZ * 1000.0,
            is_transition=False,
            pose_override=_interpolate_pose_sequence(landing_sequence, smooth),
            phase="landing",
            is_landing=True,
        )
        if landing_index == landing_steps:
            landing_frame["flight_state"] = "GROUNDED"
        frames.append(landing_frame)
        chain_source_seconds += 1.0 / PREVIEW_SAMPLE_HZ
    chain_pos = landing_end
    segment_manifest.append({
        "index": len(CHAIN_ORDER),
        "route": "landing",
        "trajectory_id": "flight_leg_posture_v3",
        "trajectory_type": "synthetic_descent_with_cached_landing_joints",
        "source_duration_ms": LANDING_SOURCE_SECONDS * 1000.0,
    })

    path = np.asarray([_scene_position(frame["fly_pos"]) for frame in frames])
    return {
        "trajectory_id": "12-segment-composite-with-landing",
        "trajectory_type": "synthetic_chain_of_measured_segments",
        "source_frames": sum(previews[name]["source_frames"] for name in CHAIN_ORDER),
        "source_duration_ms": round(
            sum(previews[name]["source_duration_ms"] for name in CHAIN_ORDER), 2
        ),
        "transition_duration_ms": round(
            (len(CHAIN_ORDER) - 1) * TRANSITION_SOURCE_SECONDS * 1000.0, 2
        ),
        "landing_duration_ms": LANDING_SOURCE_SECONDS * 1000.0,
        "preview_duration_ms": round(chain_source_seconds * 1000.0 / PLAYBACK_RATE, 2),
        "displacement_mm": round(float(np.linalg.norm(path[-1] - path[0])), 3),
        "path_length_mm": round(float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()), 3),
        "median_speed_mm_s": None,
        "composition": "measured_segments_with_synthetic_recovery_transitions_and_landing",
        "segments": segment_manifest,
        "frames": frames,
    }


def build_preview_cache() -> dict:
    import h5py

    geom_names, base_pose = _tucked_pose()
    pattern = np.load(WING_PATTERN_PATH)
    previews = {}
    with h5py.File(DATASET_PATH, "r") as dataset:
        timestep = float(dataset["timestep_seconds"][()])
        if not np.isclose(timestep, 1.0 / SOURCE_SAMPLE_HZ):
            raise ValueError(f"Unexpected flight dataset timestep: {timestep}")
        for route_name, route in ROUTES.items():
            group = dataset["trajectories"][route["trajectory_id"]]
            converted = _convert_route(group, geom_names, base_pose, pattern)
            if converted["trajectory_type"] != route["expected_type"]:
                raise ValueError(
                    f"Trajectory {route['trajectory_id']} is {converted['trajectory_type']}, "
                    f"expected {route['expected_type']}"
                )
            previews[route_name] = {**route, **converted}

    previews[CHAIN_ROUTE_NAME] = _build_maneuver_chain(
        previews, geom_names, base_pose, pattern
    )

    payload = {
        "schema_version": "flight-data-preview-v1",
        "source": {
            "dataset": str(DATASET_PATH.relative_to(PROJECT_DIR)),
            "wing_pattern": str(WING_PATTERN_PATH.relative_to(PROJECT_DIR)),
            "base_pose": str(BASE_POSE_CACHE.relative_to(PROJECT_DIR)),
            "source_sample_hz": SOURCE_SAMPLE_HZ,
            "preview_sample_hz": PREVIEW_SAMPLE_HZ,
            "playback_rate": PLAYBACK_RATE,
            "position_units": "mm",
            "policy_used": False,
        },
        "geom_names": geom_names,
        "previews": previews,
    }
    PREVIEW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_CACHE.write_text(json.dumps(payload, separators=(",", ":")))
    return payload


def load_preview_cache() -> dict:
    if not PREVIEW_CACHE.exists():
        return build_preview_cache()
    return json.loads(PREVIEW_CACHE.read_text())


class FlightDataPreviewBridge:
    def __init__(self):
        self.queue = None
        self.running = False
        self._thread = None
        self._loop = None
        self._route_name = None

    def start(self, loop, queue, route_name: str):
        if route_name not in AVAILABLE_ROUTES:
            raise ValueError(f"Unknown flight preview route: {route_name}")
        self.queue = queue
        self._loop = loop
        self._route_name = route_name
        self.running = True
        self._thread = threading.Thread(target=self._run_safe, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _emit(self, data):
        future = asyncio.run_coroutine_threadsafe(self.queue.put(data), self._loop)
        future.result(timeout=5)

    def _run_safe(self):
        try:
            self._run()
        except Exception as exc:
            import traceback
            print(f"[FlightPreview] ERROR: {exc}", flush=True)
            traceback.print_exc()
            self.running = False
            self._emit({"event": "walk_end", "preview_error": str(exc)})

    def _run(self):
        cached = load_preview_cache()
        preview = cached["previews"][self._route_name]
        source = cached["source"]
        self._emit({
            "event": "walk_init",
            "geom_names": cached["geom_names"],
            "render_mode": "flight_data_preview",
            "preview": {
                "route": self._route_name,
                "trajectory_id": preview["trajectory_id"],
                "trajectory_type": preview["trajectory_type"],
                "source_duration_ms": preview["source_duration_ms"],
                "playback_rate": source["playback_rate"],
                "source": source["dataset"],
                "composition": preview.get("composition", "single_measured_segment"),
                "segment_count": len(preview.get("segments", [])) or 1,
            },
        })
        frame_interval = 1.0 / (source["preview_sample_hz"] * source["playback_rate"])
        for cached_frame in preview["frames"]:
            if not self.running:
                break
            frame = dict(cached_frame)
            frame["event"] = "walk_frame"
            frame["preview_route"] = self._route_name
            self._emit(frame)
            time.sleep(frame_interval)
        self.running = False
        self._emit({"event": "walk_end", "preview_route": self._route_name})


if __name__ == "__main__":
    data = build_preview_cache()
    for name, preview in data["previews"].items():
        print(
            f"{name}: {len(preview['frames'])} frames, "
            f"{preview['source_duration_ms']} ms source, "
            f"{preview['displacement_mm']} mm displacement"
        )
