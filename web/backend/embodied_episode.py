"""Unified, observable multi-behavior episode playback.

Episode v1 intentionally composes existing controller caches. Its manifest and
WebSocket protocol are the stable seam that a later local closed-loop
brain/body recorder will write to without changing the browser player.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from copy import deepcopy
from pathlib import Path

try:
    from .motor_state_resolver import MotorStateResolver
except ImportError:  # Uvicorn is launched with web/backend as its working dir.
    from motor_state_resolver import MotorStateResolver


BACKEND_DIR = Path(__file__).resolve().parent
CACHE_DIR = BACKEND_DIR / "walk_cache"
EPISODE_DIR = BACKEND_DIR / "episodes"
NEURAL_TRACE_DIR = EPISODE_DIR / "traces"
SCHEMA_VERSION = "flywire-episode-v1"
NEURAL_SCHEMA_VERSION = "flywire-neural-trace-v1"


class EpisodeValidationError(ValueError):
    """Raised when an episode manifest cannot be played safely."""


def _safe_child(directory: Path, filename: str, suffix: str) -> Path:
    if Path(filename).name != filename or not filename.endswith(suffix):
        raise EpisodeValidationError(f"Unsafe episode asset name: {filename!r}")
    path = (directory / filename).resolve()
    if path.parent != directory.resolve():
        raise EpisodeValidationError(f"Episode asset escapes directory: {filename!r}")
    return path


def load_episode_manifest(episode_id: str) -> dict:
    path = _safe_child(EPISODE_DIR, f"{episode_id}.json", ".json")
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as file:
        manifest = json.load(file)
    validate_episode_manifest(manifest)
    return manifest


def validate_episode_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise EpisodeValidationError("Unsupported or missing episode schema_version")
    if not manifest.get("id") or not manifest.get("title"):
        raise EpisodeValidationError("Episode id and title are required")
    fps = manifest.get("playback_fps")
    if not isinstance(fps, (int, float)) or not 1 <= fps <= 120:
        raise EpisodeValidationError("playback_fps must be between 1 and 120")
    segments = manifest.get("segments")
    if not isinstance(segments, list) or not segments:
        raise EpisodeValidationError("Episode must contain at least one segment")

    neural_trace_name = manifest.get("neural_trace")
    if neural_trace_name:
        trace_path = _safe_child(NEURAL_TRACE_DIR, neural_trace_name, ".json")
        if not trace_path.exists():
            raise EpisodeValidationError(f"Missing neural trace: {neural_trace_name}")

    causal_motor = manifest.get("motor_control") == "dn_state_resolver"
    controller_bank = manifest.get("controller_bank", {})
    if causal_motor:
        required_controllers = {
            "idle", "walk", "backward", "groom", "groom_recover", "feed",
            "flight_takeoff", "flight", "flight_landing",
        }
        missing_controllers = required_controllers - set(controller_bank)
        if missing_controllers:
            raise EpisodeValidationError(
                f"Controller bank is missing {sorted(missing_controllers)}"
            )
        for name, spec in controller_bank.items():
            if not isinstance(spec, dict) or "cache" not in spec or "render_mode" not in spec:
                raise EpisodeValidationError(f"Invalid controller spec: {name}")
            cache_path = _safe_child(CACHE_DIR, spec["cache"], ".json")
            if not cache_path.exists():
                raise EpisodeValidationError(f"Missing controller cache: {spec['cache']}")

    seen_keys = set()
    for segment in segments:
        if causal_motor:
            required = {"key", "label", "behavior", "pipeline", "neural_clip", "duration_frames"}
        else:
            required = {"key", "label", "behavior", "cache", "render_mode", "pipeline"}
        if neural_trace_name:
            required.add("neural_clip")
        missing = required - set(segment)
        if missing:
            raise EpisodeValidationError(
                f"Segment {segment.get('key', '?')} is missing {sorted(missing)}"
            )
        if segment["key"] in seen_keys:
            raise EpisodeValidationError(f"Duplicate segment key: {segment['key']}")
        seen_keys.add(segment["key"])
        if causal_motor:
            if not isinstance(segment["duration_frames"], int) or segment["duration_frames"] < 1:
                raise EpisodeValidationError("duration_frames must be a positive integer")
        else:
            cache_path = _safe_child(CACHE_DIR, segment["cache"], ".json")
            if not cache_path.exists():
                raise EpisodeValidationError(f"Missing cache: {segment['cache']}")


def _load_neural_trace(manifest: dict) -> dict | None:
    trace_name = manifest.get("neural_trace")
    if not trace_name:
        return None
    trace_path = _safe_child(NEURAL_TRACE_DIR, trace_name, ".json")
    with trace_path.open() as file:
        trace = json.load(file)
    if trace.get("schema_version") != NEURAL_SCHEMA_VERSION:
        raise EpisodeValidationError(f"Unsupported neural trace: {trace_name}")
    clips = trace.get("clips")
    if not isinstance(clips, dict) or not clips:
        raise EpisodeValidationError(f"Neural trace has no clips: {trace_name}")
    for segment in manifest["segments"]:
        clip = clips.get(segment.get("neural_clip"))
        if not clip or not isinstance(clip.get("frames"), list) or not clip["frames"]:
            raise EpisodeValidationError(
                f"Missing neural clip: {segment.get('neural_clip', '?')}"
            )
    return trace


def list_episode_summaries() -> list[dict]:
    summaries = []
    for path in sorted(EPISODE_DIR.glob("*.json")):
        try:
            manifest = load_episode_manifest(path.stem)
            compiled = compile_episode(manifest)
        except (EpisodeValidationError, FileNotFoundError, json.JSONDecodeError):
            continue
        summaries.append(compiled["metadata"])
    return summaries


def _load_segment_cache(segment: dict) -> tuple[list[str], list[dict]]:
    cache_path = _safe_child(CACHE_DIR, segment["cache"], ".json")
    with cache_path.open() as file:
        cache = json.load(file)
    geom_names = cache.get("geom_names")
    frames = cache.get("frames")
    if not isinstance(geom_names, list) or not isinstance(frames, list) or not frames:
        raise EpisodeValidationError(f"Invalid cache payload: {segment['cache']}")

    start = int(segment.get("start_frame", 0))
    end = int(segment.get("end_frame", len(frames)))
    if not 0 <= start < end <= len(frames):
        raise EpisodeValidationError(
            f"Invalid frame range {start}:{end} for {segment['cache']} ({len(frames)})"
        )
    return geom_names, frames[start:end]


def _load_controller_bank(manifest: dict) -> dict:
    controllers = {}
    for name, spec in manifest["controller_bank"].items():
        cache_path = _safe_child(CACHE_DIR, spec["cache"], ".json")
        with cache_path.open() as file:
            cache = json.load(file)
        frames = cache.get("frames")
        geom_names = cache.get("geom_names")
        if not isinstance(frames, list) or not frames or not isinstance(geom_names, list):
            raise EpisodeValidationError(f"Invalid controller cache: {spec['cache']}")
        start = int(spec.get("start_frame", 0))
        end = int(spec.get("end_frame", len(frames)))
        if not 0 <= start < end <= len(frames):
            raise EpisodeValidationError(
                f"Invalid controller range {start}:{end} for {name}"
            )
        controllers[name] = {
            **deepcopy(spec),
            "geom_names": geom_names,
            "frames": frames[start:end],
            "loop_start_index": int(spec.get("loop_start_frame", start)) - start,
        }
        if not 0 <= controllers[name]["loop_start_index"] < len(controllers[name]["frames"]):
            raise EpisodeValidationError(f"Invalid loop_start_frame for {name}")
    return controllers


def _neural_payload(neural_trace: dict, segment: dict, source: dict) -> dict:
    return {
        "source": "recorded_lif",
        "trace_id": neural_trace["id"],
        "clip_id": segment["neural_clip"],
        "source_t_ms": source["t_ms"],
        "stimulus": source.get("stimulus", segment["neural_clip"]),
        "behavior_mode": source["behavior_mode"],
        "brain_steps": source["brain_steps"],
        "total_spikes": source["total_spikes"],
        "dn": deepcopy(source["dn"]),
        "pop": deepcopy(source["pop"]),
    }


def _compile_causal_episode(manifest: dict, neural_trace: dict) -> dict:
    """Let recorded DN rates choose the body controller on every frame."""
    fps = float(manifest["playback_fps"])
    dt_ms = 1000.0 / fps
    resolver = MotorStateResolver()
    controllers = _load_controller_bank(manifest)
    controller_cursors = {name: 0 for name in controllers}
    compiled_segments = []
    timeline_frames = []
    world_position = [0.0, 0.0, 0.0]
    last_source_position = None
    active_controller = None
    global_frame = 0

    for segment_index, segment in enumerate(manifest["segments"]):
        neural_frames = neural_trace["clips"][segment["neural_clip"]]["frames"]
        frame_count = segment["duration_frames"]
        segment_start = global_frame
        start_ms = global_frame / fps * 1000.0

        for local_index in range(frame_count):
            progress = local_index / max(1, frame_count - 1)
            neural_index = round(progress * (len(neural_frames) - 1))
            neural_source = neural_frames[neural_index]
            neural = _neural_payload(neural_trace, segment, neural_source)
            decision = resolver.update(neural["dn"], dt_ms)
            controller_name = decision.controller
            controller = controllers[controller_name]
            controller_changed = controller_name != active_controller

            if controller_changed:
                controller_cursors[controller_name] = 0
                active_controller = controller_name
                last_source_position = None

            cursor = controller_cursors[controller_name]
            source_frame = controller["frames"][cursor]
            source_position = [
                float(value) for value in source_frame.get("fly_pos", [0.0, 0.0, 0.0])
            ]
            if last_source_position is not None and not controller.get("stationary", False):
                for axis in range(3):
                    world_position[axis] += source_position[axis] - last_source_position[axis]
            if controller["render_mode"] == "flight_path":
                world_position[2] = max(0.0, world_position[2])
            else:
                world_position[2] = 0.0
            last_source_position = source_position

            next_cursor = cursor + 1
            if next_cursor >= len(controller["frames"]):
                next_cursor = controller["loop_start_index"]
            controller_cursors[controller_name] = next_cursor
            if next_cursor <= cursor:
                last_source_position = None

            frame = deepcopy(source_frame)
            frame["fly_pos"] = [round(value, 3) for value in world_position]
            frame["episode_t_ms"] = round(global_frame / fps * 1000.0, 1)
            frame["source_t_ms"] = frame.get("t_ms", 0.0)
            frame["t_ms"] = frame["episode_t_ms"]
            frame["episode_id"] = manifest["id"]
            frame["episode_segment"] = segment["key"]
            frame["episode_segment_index"] = segment_index
            frame["episode_local_frame"] = local_index
            frame["behavior_mode"] = decision.behavior_intent
            frame["flight_state"] = decision.flight_state
            frame["neural"] = neural
            frame["motor_decision"] = {
                **decision.to_dict(),
                "controller_changed": controller_changed,
            }
            if controller_changed:
                frame["motor_init"] = {
                    "geom_names": controller["geom_names"],
                    "render_mode": controller["render_mode"],
                    "controller": controller_name,
                }
            timeline_frames.append(frame)
            global_frame += 1

        compiled_segments.append({
            "key": segment["key"],
            "label": segment["label"],
            "behavior": segment["behavior"],
            "pipeline": segment["pipeline"],
            "neural_clip": segment["neural_clip"],
            "frame_count": frame_count,
            "start_frame": segment_start,
            "end_frame": global_frame,
            "start_ms": round(start_ms, 1),
            "duration_ms": round(frame_count / fps * 1000.0, 1),
        })

    metadata = {
        "schema_version": manifest["schema_version"],
        "id": manifest["id"],
        "title": manifest["title"],
        "description": manifest.get("description", ""),
        "source_mode": manifest["source_mode"],
        "motor_control": manifest["motor_control"],
        "playback_fps": fps,
        "frame_count": global_frame,
        "duration_ms": round(global_frame / fps * 1000.0, 1),
        "segments": deepcopy(compiled_segments),
        "neural_source": {
            "trace_id": neural_trace["id"],
            **deepcopy(neural_trace["provenance"]),
            "coupling": "causal_motor_selection",
            "causal_body_control": True,
        },
    }
    return {"metadata": metadata, "segments": compiled_segments, "frames": timeline_frames}


def compile_episode(manifest: dict) -> dict:
    """Resolve cache references into a monotonic, position-continuous timeline."""
    validate_episode_manifest(manifest)
    neural_trace = _load_neural_trace(manifest)
    if manifest.get("motor_control") == "dn_state_resolver":
        if neural_trace is None:
            raise EpisodeValidationError("Causal motor episode requires a neural trace")
        return _compile_causal_episode(manifest, neural_trace)
    fps = float(manifest["playback_fps"])
    compiled_segments = []
    timeline_frames = []
    global_frame = 0
    last_position = [0.0, 0.0, 0.0]

    for segment_index, segment in enumerate(manifest["segments"]):
        geom_names, source_frames = _load_segment_cache(segment)
        neural_frames = None
        if neural_trace:
            neural_frames = neural_trace["clips"][segment["neural_clip"]]["frames"]
        first_position = source_frames[0].get("fly_pos", [0.0, 0.0, 0.0])
        offset = [last_position[i] - float(first_position[i]) for i in range(3)]
        start_ms = global_frame / fps * 1000.0

        for local_index, source_frame in enumerate(source_frames):
            frame = deepcopy(source_frame)
            local_position = frame.get("fly_pos", [0.0, 0.0, 0.0])
            frame["fly_pos"] = [
                round(float(local_position[i]) + offset[i], 3) for i in range(3)
            ]
            frame["episode_t_ms"] = round(global_frame / fps * 1000.0, 1)
            frame["source_t_ms"] = frame.get("t_ms", 0.0)
            frame["t_ms"] = frame["episode_t_ms"]
            frame["episode_id"] = manifest["id"]
            frame["episode_segment"] = segment["key"]
            frame["episode_segment_index"] = segment_index
            frame["episode_local_frame"] = local_index
            if neural_frames:
                progress = local_index / max(1, len(source_frames) - 1)
                neural_index = round(progress * (len(neural_frames) - 1))
                neural_source = neural_frames[neural_index]
                frame["neural"] = _neural_payload(neural_trace, segment, neural_source)
            timeline_frames.append(frame)
            last_position = frame["fly_pos"]
            global_frame += 1

        compiled_segments.append({
            "key": segment["key"],
            "label": segment["label"],
            "behavior": segment["behavior"],
            "render_mode": segment["render_mode"],
            "pipeline": segment["pipeline"],
            "neural_clip": segment.get("neural_clip"),
            "geom_names": geom_names,
            "frame_count": len(source_frames),
            "start_frame": global_frame - len(source_frames),
            "end_frame": global_frame,
            "start_ms": round(start_ms, 1),
            "duration_ms": round(len(source_frames) / fps * 1000.0, 1),
        })

    duration_ms = global_frame / fps * 1000.0
    metadata = {
        "schema_version": manifest["schema_version"],
        "id": manifest["id"],
        "title": manifest["title"],
        "description": manifest.get("description", ""),
        "source_mode": manifest.get("source_mode", "unknown"),
        "playback_fps": fps,
        "frame_count": global_frame,
        "duration_ms": round(duration_ms, 1),
        "segments": [
            {key: value for key, value in segment.items() if key != "geom_names"}
            for segment in compiled_segments
        ],
    }
    if neural_trace:
        metadata["neural_source"] = {
            "trace_id": neural_trace["id"],
            **deepcopy(neural_trace["provenance"]),
            "coupling": "synchronized_overlay",
            "causal_body_control": False,
        }
    return {
        "metadata": metadata,
        "segments": compiled_segments,
        "frames": timeline_frames,
    }


class EmbodiedEpisodeBridge:
    """Stream a compiled episode through the existing body WebSocket channel."""

    def __init__(self):
        self.queue = None
        self._loop = None
        self._thread = None
        self.running = False

    def start(self, loop, queue, episode_id="embodied_demo_v1"):
        if self.running:
            return
        self.queue = queue
        self._loop = loop
        self.running = True
        self._thread = threading.Thread(
            target=self._run_safe, args=(episode_id,), daemon=True
        )
        self._thread.start()

    def stop(self):
        self.running = False

    def _emit(self, data):
        try:
            future = asyncio.run_coroutine_threadsafe(self.queue.put(data), self._loop)
            future.result(timeout=5)
        except Exception as exc:
            print(f"[Episode] emit error: {exc}", flush=True)

    def _run_safe(self, episode_id):
        try:
            self._run(episode_id)
        except Exception as exc:
            import traceback
            print(f"[Episode] ERROR: {exc}", flush=True)
            traceback.print_exc()
            self.running = False
            self._emit({"event": "episode_end", "completed": False, "error": str(exc)})
            self._emit({"event": "walk_end"})

    def _run(self, episode_id):
        compiled = compile_episode(load_episode_manifest(episode_id))
        metadata = compiled["metadata"]
        fps = metadata["playback_fps"]
        total_frames = metadata["frame_count"]
        print(
            f"[Episode] {episode_id}: {total_frames} frames, "
            f"{metadata['duration_ms'] / 1000:.1f}s",
            flush=True,
        )
        self._emit({"event": "episode_init", **metadata})

        emitted_frames = 0
        completed = True
        for segment_index, segment in enumerate(compiled["segments"]):
            if not self.running:
                completed = False
                break
            self._emit({
                "event": "episode_segment",
                "episode_id": episode_id,
                "segment_index": segment_index,
                "segment_count": len(compiled["segments"]),
                "key": segment["key"],
                "label": segment["label"],
                "behavior": segment["behavior"],
                "pipeline": segment["pipeline"],
                "start_ms": segment["start_ms"],
                "duration_ms": segment["duration_ms"],
                "motor_control": metadata.get("motor_control"),
            })
            if metadata.get("motor_control") != "dn_state_resolver":
                self._emit({
                    "event": "walk_init",
                    "geom_names": segment["geom_names"],
                    "render_mode": segment["render_mode"],
                    "episode_id": episode_id,
                })

            start, end = segment["start_frame"], segment["end_frame"]
            for frame in compiled["frames"][start:end]:
                if not self.running:
                    completed = False
                    break
                payload = deepcopy(frame)
                motor_init = payload.pop("motor_init", None)
                if motor_init:
                    self._emit({
                        "event": "walk_init",
                        "episode_id": episode_id,
                        **motor_init,
                    })
                payload["event"] = "walk_frame"
                payload["episode_progress"] = round(
                    (emitted_frames + 1) / total_frames, 5
                )
                self._emit(payload)
                emitted_frames += 1
                time.sleep(1.0 / fps)
            if not self.running:
                break

        self.running = False
        self._emit({
            "event": "episode_end",
            "episode_id": episode_id,
            "completed": completed,
            "emitted_frames": emitted_frames,
            "frame_count": total_frames,
        })
        self._emit({"event": "walk_end", "episode_id": episode_id})
