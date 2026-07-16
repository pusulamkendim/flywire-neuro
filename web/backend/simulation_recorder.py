"""Compressed, lossless event recording for every simulation run."""

from __future__ import annotations

import asyncio
import gzip
import json
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RECORDINGS_DIR = Path(__file__).resolve().parent / "simulation_records"
TERMINAL_EVENTS = {"end", "walk_end", "episode_end"}


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _normalise_payload(item: Any) -> dict:
    if item is None:
        return {"event": "end"}
    if hasattr(item, "model_dump"):
        item = item.model_dump()
    if not isinstance(item, dict):
        return {"event": "unknown", "value": item}
    payload = dict(item)
    payload.setdefault("event", "frame")
    return payload


class SimulationRecorder:
    """Own one Start-to-Stop recording and its inspectable manifest."""

    def __init__(self, directory: Path = RECORDINGS_DIR):
        self.directory = directory
        self._lock = threading.RLock()
        self._active = None
        self._stream = None
        self._requested_status = None
        self.last_completed_id = None
        # JSON encoding and gzip compression are intentionally kept off the
        # asyncio/render timing path. A single writer preserves event order.
        self._writer = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="simulation-recorder"
        )

    @property
    def current_id(self) -> str | None:
        with self._lock:
            return self._active["id"] if self._active else None

    def start(self, simulation_type: str, config: dict | None = None) -> str:
        return self._writer.submit(
            self._start_sync, simulation_type, config
        ).result()

    def _start_sync(self, simulation_type: str, config: dict | None = None) -> str:
        with self._lock:
            if self._active:
                self._finish_sync("superseded")
            self.directory.mkdir(parents=True, exist_ok=True)
            now = datetime.now(timezone.utc)
            safe_type = re.sub(r"[^a-zA-Z0-9_-]+", "-", simulation_type).strip("-")
            recording_id = f"{now.strftime('%Y%m%d_%H%M%S_%f')[:-3]}_{safe_type}"
            run_dir = self.directory / recording_id
            run_dir.mkdir(parents=False, exist_ok=False)
            data_path = run_dir / "frames.jsonl.gz"
            self._stream = gzip.open(
                data_path, "wt", encoding="utf-8", compresslevel=3
            )
            self._active = {
                "schema_version": "flywire-simulation-recording-v1",
                "id": recording_id,
                "simulation_type": simulation_type,
                "config": config or {},
                "status": "running",
                "started_at_utc": now.isoformat(),
                "ended_at_utc": None,
                "duration_ms": None,
                "record_count": 0,
                "event_counts": {},
                "data_file": "frames.jsonl.gz",
                "compression": "gzip",
                "bytes_compressed": 0,
                "timing_schema": {
                    "recorded_elapsed_ms": "monotonic wall time since recording start",
                    "clock.simulation_time_ms": "producer simulation timeline",
                    "clock.body_runtime_ms": "real-time 30 Hz Digital Life body timeline",
                    "clock.body_simulation_ms": "motor timeline advanced by wall time or computed LIF time",
                    "clock.brain_simulation_ms": "138K LIF model time, not wall time",
                    "clock.source_simulation_ms": "source mocap/physics dataset time",
                    "clock.playback_timeline_ms": "resampled or slowed playback time",
                },
                "_started_monotonic": time.monotonic(),
                "_run_dir": run_dir,
                "_event_counter": Counter(),
            }
            self._requested_status = None
            self._write_manifest()
            self._write_record({
                "event": "recording_start",
                "simulation_type": simulation_type,
                "config": config or {},
            })
            return recording_id

    def request_stop(self):
        with self._lock:
            if self._active:
                self._requested_status = "stopped"

    def record(self, item: Any):
        payload = _normalise_payload(item)
        with self._lock:
            if not self._active:
                return None
            elapsed_ms = (
                time.monotonic() - self._active["_started_monotonic"]
            ) * 1000.0
            recorded_at = datetime.now(timezone.utc).isoformat()
        future = self._writer.submit(
            self._record_sync, payload, elapsed_ms, recorded_at
        )
        future.add_done_callback(self._report_writer_error)
        return future

    @staticmethod
    def _report_writer_error(future):
        error = future.exception()
        if error is not None:
            print(f"[Recorder] ERROR: {error}", flush=True)

    def _record_sync(self, payload: dict, elapsed_ms: float, recorded_at: str):
        with self._lock:
            if not self._active or not self._stream:
                return
            self._write_record(payload, elapsed_ms, recorded_at)
            event = payload.get("event", "frame")
            if event in TERMINAL_EVENTS:
                has_error = any(str(key).endswith("_error") for key in payload)
                status = self._requested_status or ("error" if has_error else "completed")
                self._finish_sync(status)

    def _write_record(
        self,
        payload: dict,
        elapsed_ms: float | None = None,
        recorded_at: str | None = None,
    ):
        if elapsed_ms is None:
            elapsed_ms = (
                time.monotonic() - self._active["_started_monotonic"]
            ) * 1000.0
        clock = self._clock_fields(payload, elapsed_ms)
        row = {
            "record_index": self._active["record_count"],
            "recorded_elapsed_ms": round(elapsed_ms, 3),
            "recorded_at_utc": recorded_at or datetime.now(timezone.utc).isoformat(),
            "clock": clock,
            **payload,
        }
        self._stream.write(json.dumps(
            row, separators=(",", ":"), default=_json_default, allow_nan=False
        ))
        self._stream.write("\n")
        self._active["record_count"] += 1
        self._active["_event_counter"][row.get("event", "frame")] += 1
        if self._active["record_count"] % 30 == 0:
            self._stream.flush()

    def _clock_fields(self, payload: dict, wall_elapsed_ms: float) -> dict:
        clock = {"wall_elapsed_ms": round(wall_elapsed_ms, 3)}
        event = payload.get("event")
        producer_t = payload.get("t_ms")
        source_t = payload.get("source_t_ms")
        brain_t = payload.get("brain_t_ms")
        body_simulation_t = payload.get("body_simulation_ms")
        simulation_type = self._active["simulation_type"]

        if simulation_type == "digital-life":
            if event == "brain_frame" and producer_t is not None:
                clock["brain_simulation_ms"] = producer_t
            elif event == "walk_frame":
                if producer_t is not None:
                    clock["body_runtime_ms"] = producer_t
                if brain_t is not None:
                    clock["brain_simulation_ms"] = brain_t
                if body_simulation_t is not None:
                    clock["body_simulation_ms"] = body_simulation_t
        elif source_t is not None:
            clock["source_simulation_ms"] = source_t
            if producer_t is not None:
                clock["playback_timeline_ms"] = producer_t
        elif producer_t is not None:
            clock["simulation_time_ms"] = producer_t
        return clock

    def finish(self, status: str = "stopped") -> str | None:
        return self._writer.submit(self._finish_sync, status).result()

    def _finish_sync(self, status: str = "stopped") -> str | None:
        with self._lock:
            if not self._active:
                return self.last_completed_id
            now = datetime.now(timezone.utc)
            self._active["status"] = status
            self._active["ended_at_utc"] = now.isoformat()
            self._active["duration_ms"] = round(
                (time.monotonic() - self._active["_started_monotonic"]) * 1000.0, 3
            )
            self._active["event_counts"] = dict(self._active["_event_counter"])
            self._stream.flush()
            self._stream.close()
            self._stream = None
            data_path = self._active["_run_dir"] / self._active["data_file"]
            self._active["bytes_compressed"] = data_path.stat().st_size
            recording_id = self._active["id"]
            self._write_manifest()
            self.last_completed_id = recording_id
            self._active = None
            self._requested_status = None
            return recording_id

    def _public_manifest(self, active: dict | None = None) -> dict:
        source = active or self._active
        if not source:
            return {}
        manifest = {
            key: value for key, value in source.items() if not key.startswith("_")
        }
        if source.get("_event_counter") is not None:
            manifest["event_counts"] = dict(source["_event_counter"])
        if manifest.get("status") == "running":
            manifest["duration_ms"] = round(
                (time.monotonic() - source["_started_monotonic"]) * 1000.0, 3
            )
        return manifest

    def _write_manifest(self):
        manifest_path = self._active["_run_dir"] / "manifest.json"
        manifest_path.write_text(
            json.dumps(self._public_manifest(), indent=2, default=_json_default)
        )

    def list(self, limit: int = 100) -> list[dict]:
        with self._lock:
            if not self.directory.exists():
                return []
            results = []
            for path in sorted(self.directory.glob("*/manifest.json"), reverse=True):
                try:
                    manifest = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if self._active and manifest.get("id") == self._active["id"]:
                    manifest = self._public_manifest()
                results.append(manifest)
                if len(results) >= limit:
                    break
            return results

    def get(self, recording_id: str) -> dict:
        path = self._safe_run_dir(recording_id) / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(recording_id)
        if self._active and recording_id == self._active["id"]:
            return self._public_manifest()
        return json.loads(path.read_text())

    def data_path(self, recording_id: str) -> Path:
        path = self._safe_run_dir(recording_id) / "frames.jsonl.gz"
        if not path.exists():
            raise FileNotFoundError(recording_id)
        return path

    def _safe_run_dir(self, recording_id: str) -> Path:
        if Path(recording_id).name != recording_id:
            raise FileNotFoundError(recording_id)
        run_dir = (self.directory / recording_id).resolve()
        if run_dir.parent != self.directory.resolve():
            raise FileNotFoundError(recording_id)
        return run_dir


class RecordingQueue(asyncio.Queue):
    """Queue that records exactly the payload later sent over WebSocket."""

    def __init__(self, recorder: SimulationRecorder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder = recorder

    async def put(self, item):
        try:
            self.recorder.record(item)
        except Exception as exc:
            # Recording must never stop the simulation or WebSocket stream.
            print(f"[Recorder] ERROR: {exc}", flush=True)
            self.recorder.finish("error")
        await super().put(item)
