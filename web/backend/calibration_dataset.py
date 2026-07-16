"""Build reusable sensory-response datasets and calibration trial plans.

This script does not pretend the legacy recording is a balanced calibration.
`seed` converts available real LIF blocks to the new sensory contract, while
`plan` enumerates the intensity/pulse trials a future local 138K run must fill.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from .world_config import BACKEND_DIR, load_sensory_contract
except ImportError:
    from world_config import BACKEND_DIR, load_sensory_contract


DATASET_DIR = BACKEND_DIR / "datasets"
DN_KEYS = ("forward", "turn_L", "turn_R", "backward", "escape", "groom", "feed")
POP_KEYS = (
    "pam", "ppl1", "mbon_approach", "mbon_avoidance", "mbon_suppress",
    "serotonin", "octopamine", "gaba", "ach", "glut",
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_plan(repeats=3) -> dict:
    contract = load_sensory_contract()
    trials = []
    for channel_name, channel in contract["channels"].items():
        brain_stimulus = channel.get("brain_stimulus")
        if not brain_stimulus:
            continue
        calibration = channel["calibration"]
        for intensity in calibration["intensities"]:
            for pulse_ms in calibration["pulse_ms"]:
                for repeat in range(1, repeats + 1):
                    trials.append({
                        "trial_id": f"{channel_name}_i{int(intensity * 100):03d}_p{pulse_ms}_r{repeat}",
                        "sensory_channel": channel_name,
                        "brain_stimulus": brain_stimulus,
                        "intensity": intensity,
                        "baseline_ms": 500,
                        "pulse_ms": pulse_ms,
                        "recovery_ms": 1000,
                        "repeat": repeat,
                        "status": "planned",
                    })
    simulated_ms = sum(
        trial["baseline_ms"] + trial["pulse_ms"] + trial["recovery_ms"]
        for trial in trials
    )
    return {
        "schema_version": "flywire-dn-calibration-plan-v1",
        "id": "dn_calibration_v1_plan",
        "created_at": _timestamp(),
        "sensory_contract_id": contract["id"],
        "status": "planned",
        "repeats": repeats,
        "trial_count": len(trials),
        "simulated_duration_ms": simulated_ms,
        "notes": [
            "Run locally; the web server consumes recorded responses only.",
            "Keep the full brain state continuous within a trial, then restore a seeded baseline.",
            "Record individual DN rates as well as grouped normalized outputs.",
        ],
        "trials": trials,
    }


def _contiguous_blocks(rows: list[dict], stimulus: str) -> list[list[dict]]:
    blocks = []
    current = []
    for row in rows:
        if row["stimuli"] == stimulus:
            current.append(row)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def build_seed(source_log: Path) -> dict:
    contract = load_sensory_contract()
    rows = list(csv.DictReader(source_log.open()))
    responses = {}
    for channel_name, channel in contract["channels"].items():
        stimulus = channel.get("brain_stimulus")
        if not stimulus:
            continue
        blocks = _contiguous_blocks(rows, stimulus)
        if not blocks:
            continue
        block = max(blocks, key=len)
        t0 = float(block[0]["t_ms"])
        frames = []
        for row in block:
            frames.append({
                "t_ms": round(float(row["t_ms"]) - t0, 3),
                "total_spikes": int(row["total_spikes"]),
                "dn": {key: float(row[f"dn_{key}"]) for key in DN_KEYS},
                "pop": {key: int(row[f"pop_{key}"]) for key in POP_KEYS},
            })
        responses[channel_name] = {
            "sensory_channel": channel_name,
            "brain_stimulus": stimulus,
            "intensity": 1.0,
            "source": "legacy_continuous_lif_block",
            "calibrated": False,
            "frames": frames,
        }
    return {
        "schema_version": "flywire-dn-response-library-v1",
        "id": "dn_response_seed_v1",
        "created_at": _timestamp(),
        "status": "seed_unbalanced",
        "sensory_contract_id": contract["id"],
        "provenance": {
            "kind": "recorded_lif",
            "source_log": source_log.name,
            "neuron_count": 138639,
            "warning": "Continuous legacy run with carry-over activity; not an intensity calibration.",
        },
        "available_channels": sorted(responses),
        "responses": responses,
    }


def write_dataset(payload: dict, filename: str) -> Path:
    DATASET_DIR.mkdir(exist_ok=True)
    path = DATASET_DIR / filename
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--repeats", type=int, default=3)
    seed = sub.add_parser("seed")
    seed.add_argument(
        "--source-log",
        type=Path,
        default=BACKEND_DIR / "brain_logs" / "brain_20260316_161125.csv",
    )
    args = parser.parse_args()
    if args.command == "plan":
        payload = build_plan(args.repeats)
        path = write_dataset(payload, "dn_calibration_v1_plan.json")
    else:
        payload = build_seed(args.source_log)
        path = write_dataset(payload, "dn_response_seed_v1.json")
    print(f"{path}: {payload.get('trial_count', len(payload.get('responses', {})))} records")


if __name__ == "__main__":
    main()
