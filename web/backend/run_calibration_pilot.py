"""Run a real batched 138K-neuron intensity sweep on the local machine.

Four stimulus intensities are simulated in parallel for each mapped sensory
channel. The output is a new DN response dataset, not a reformat of old logs.
This pilot fixes pulse duration at 250 ms and uses one seeded repeat; the full
288-trial protocol remains the next, longer calibration run.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent.parent
FLY_BRAIN_DIR = PROJECT_DIR / "fly-brain-embodied"
CODE_DIR = FLY_BRAIN_DIR / "code"
DATA_DIR = FLY_BRAIN_DIR / "data"
for path in (CODE_DIR, FLY_BRAIN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import benchmark  # noqa: E402

benchmark.COMP_PATH = str(DATA_DIR / "2025_Completeness_783.csv")
benchmark.CONN_PATH = str(DATA_DIR / "2025_Connectivity_783.parquet")
benchmark.DATA_DIR = str(DATA_DIR)

import torch  # noqa: E402
from brain_body_bridge import DN_GROUPS, DN_NEURONS, STIMULI  # noqa: E402
from run_pytorch import DT, MODEL_PARAMS, TorchModel, get_hash_tables, get_weights  # noqa: E402

try:
    from .world_config import load_sensory_contract
except ImportError:
    from world_config import load_sensory_contract


INTENSITIES = (0.25, 0.5, 0.75, 1.0)
BASELINE_MS = 20
PULSE_MS = 50
RECOVERY_MS = 50
EMIT_MS = 1.0
WINDOW_MS = 50.0
MAX_RATE_HZ = 200.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run() -> dict:
    torch.set_num_threads(10)
    torch.manual_seed(4701)
    contract = load_sensory_contract()
    channels = [
        (name, config)
        for name, config in contract["channels"].items()
        if config.get("brain_stimulus")
    ]

    comp_path = str(DATA_DIR / "2025_Completeness_783.csv")
    conn_path = str(DATA_DIR / "2025_Connectivity_783.parquet")
    print("[Calibration] Loading FlyWire v783 weights...", flush=True)
    started = time.time()
    flyid2i, _ = get_hash_tables(comp_path)
    weights = get_weights(conn_path, comp_path, str(DATA_DIR)).to(device="cpu")
    num_neurons = len(flyid2i)
    batch = len(INTENSITIES)
    model = TorchModel(
        batch=batch,
        size=num_neurons,
        dt=DT,
        params=MODEL_PARAMS,
        weights=weights,
        device="cpu",
    )
    print(f"[Calibration] {num_neurons:,} neurons loaded in {time.time()-started:.1f}s", flush=True)

    dn_names = list(DN_NEURONS)
    dn_indices = torch.tensor([flyid2i[DN_NEURONS[name]] for name in dn_names])
    group_columns = {
        group: torch.tensor([dn_names.index(name) for name in names])
        for group, names in DN_GROUPS.items()
    }
    window_steps = int(WINDOW_MS / DT)
    emit_steps = int(EMIT_MS / DT)
    total_ms = BASELINE_MS + PULSE_MS + RECOVERY_MS
    total_steps = int(total_ms / DT)
    pulse_start = int(BASELINE_MS / DT)
    pulse_end = int((BASELINE_MS + PULSE_MS) / DT)
    responses = {}
    checkpoint_path = BACKEND_DIR / "datasets" / "dn_calibration_pilot_v1.json"
    checkpoint_path.parent.mkdir(exist_ok=True)

    with torch.no_grad():
        for channel_index, (channel_name, channel) in enumerate(channels):
            stimulus_name = channel["brain_stimulus"]
            stimulus = STIMULI[stimulus_name]
            stimulus_indices = torch.tensor([
                flyid2i[body_id] for body_id in stimulus["neurons"] if body_id in flyid2i
            ])
            rates = torch.zeros(batch, num_neurons)
            scaled_rates = torch.tensor(INTENSITIES).unsqueeze(1) * float(stimulus["rate"])
            state = model.state_init()
            ring = torch.zeros(batch, window_steps, len(dn_names))
            rolling = torch.zeros(batch, len(dn_names))
            ring_cursor = 0
            frames_by_batch = [[] for _ in INTENSITIES]
            generator = torch.Generator(device="cpu")
            generator.manual_seed(4701 + channel_index)
            trial_started = time.time()

            for step in range(total_steps):
                if step == pulse_start:
                    rates[:, stimulus_indices] = scaled_rates
                elif step == pulse_end:
                    rates.zero_()

                state = model(rates, *state, generator=generator)
                spikes = state[2]
                dn_spikes = spikes[:, dn_indices]
                rolling -= ring[:, ring_cursor, :]
                ring[:, ring_cursor, :] = dn_spikes
                rolling += dn_spikes
                ring_cursor = (ring_cursor + 1) % window_steps
                if (step + 1) % emit_steps != 0:
                    continue
                t_ms = (step + 1) * DT
                samples = min(step + 1, window_steps)
                actual_window_s = samples * DT / 1000.0
                individual_hz = rolling / actual_window_s
                individual_norm = torch.clamp(individual_hz / MAX_RATE_HZ, 0, 1)
                if t_ms <= BASELINE_MS:
                    phase = "baseline"
                elif t_ms <= BASELINE_MS + PULSE_MS:
                    phase = "pulse"
                else:
                    phase = "recovery"

                for batch_index, intensity in enumerate(INTENSITIES):
                    grouped = {
                        group: round(
                            individual_norm[batch_index, columns].mean().item(), 5
                        )
                        for group, columns in group_columns.items()
                    }
                    individual = {
                        name: round(individual_norm[batch_index, i].item(), 5)
                        for i, name in enumerate(dn_names)
                    }
                    frames_by_batch[batch_index].append({
                        "t_ms": round(t_ms, 1),
                        "phase": phase,
                        "dn": grouped,
                        "dn_individual": individual,
                    })
            responses[channel_name] = {
                "sensory_channel": channel_name,
                "brain_stimulus": stimulus_name,
                "base_rate_hz": stimulus["rate"],
                "trials": [
                    {
                        "trial_id": f"{channel_name}_i{int(intensity*100):03d}_p{PULSE_MS}_r1",
                        "intensity": intensity,
                        "input_rate_hz": stimulus["rate"] * intensity,
                        "baseline_ms": BASELINE_MS,
                        "pulse_ms": PULSE_MS,
                        "recovery_ms": RECOVERY_MS,
                        "seed": 4701 + channel_index,
                        "frames": frames_by_batch[index],
                    }
                    for index, intensity in enumerate(INTENSITIES)
                ],
            }
            print(
                f"[Calibration] {channel_name}: 4 intensities × {total_ms}ms "
                f"in {time.time()-trial_started:.1f}s",
                flush=True,
            )
            checkpoint = _build_payload(
                contract_id=contract["id"],
                neuron_count=num_neurons,
                responses=responses,
                elapsed_s=time.time() - started,
                status="running_checkpoint",
            )
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2) + "\n")

    elapsed = time.time() - started
    return _build_payload(
        contract_id=contract["id"],
        neuron_count=num_neurons,
        responses=responses,
        elapsed_s=elapsed,
        status="pilot_calibrated_single_repeat",
    )


def _build_payload(contract_id, neuron_count, responses, elapsed_s, status) -> dict:
    return {
        "schema_version": "flywire-dn-response-library-v1",
        "id": "dn_calibration_pilot_v1",
        "created_at": _utc_now(),
        "status": status,
        "sensory_contract_id": contract_id,
        "provenance": {
            "kind": "new_local_lif_calibration",
            "flywire_version": "v783",
            "neuron_count": neuron_count,
            "dt_ms": DT,
            "decoder_window_ms": WINDOW_MS,
            "normalization_max_rate_hz": MAX_RATE_HZ,
            "wall_time_s": round(elapsed_s, 1),
        },
        "protocol": {
            "intensities": list(INTENSITIES),
            "baseline_ms": BASELINE_MS,
            "pulse_ms": PULSE_MS,
            "recovery_ms": RECOVERY_MS,
            "repeats": 1,
            "batch_mode": "four_intensities_parallel",
        },
        "available_channels": sorted(responses),
        "trial_count": sum(len(response["trials"]) for response in responses.values()),
        "responses": responses,
    }


def main() -> None:
    payload = run()
    output = BACKEND_DIR / "datasets" / "dn_calibration_pilot_v1.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[Calibration] Saved {output} ({output.stat().st_size/1_000_000:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
